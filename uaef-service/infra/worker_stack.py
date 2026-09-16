# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""WorkerStack — the Worker Lambda packaged with the pinned ``uaef[server]``.

The Worker Lambda is the UAEF Service's dedicated compute path and the *only*
component that imports the published library (Requirements 6.1, 6.2). It is
async-invoked (``InvocationType='Event'``) by the API Lambda, runs
``uaef.api.evaluate`` / ``batch_evaluate``, and records the outcome on the Job
record.

Single Source of Truth (Requirements 2.1–2.3):
    This stack packages the Worker by ``pip install``-ing the exact pinned spec
    ``uaef[server]==X.Y.Z`` (passed in as ``uaef_dependency`` from
    :data:`infra.constants.UAEF_PINNED_DEPENDENCY`). It bundles NO vendored or
    re-implemented UAEF source — only the service's own handler code
    (``handlers/`` + the root ``job_state``/``schemas``/``validation`` modules)
    is copied alongside the registry-installed package. There is exactly one
    place the pin is declared, and this stack consumes it.

Packaging strategy (Requirement 6.3):
    The default packaging is a **zip-asset Lambda**: CDK bundles the asset by
    installing ``uaef[server]`` into the deployment artifact. ``uaef[server]``
    is curated to stay under Lambda's 250 MB unzipped limit. IF that curated
    set ever exceeds 250 MB unzipped, the documented escape hatch is a
    **container-image Lambda** (10 GB image limit) built from
    ``docker/worker/Dockerfile`` — opt in with
    ``cdk deploy -c worker_container_image=true``. The container path installs
    the *same* pinned spec, so the single-source-of-truth contract is preserved.

Invocation strategy (Requirements 6.4, 6.5):
    * Default (Req 6.5): the API Lambda **directly** async-invokes this function
      for evaluations expected to finish within Lambda's 15-minute limit.
    * Escalation (Req 6.4): a batch whose expected execution time exceeds the
      15-minute Lambda limit must be orchestrated via **Step Functions**, fanned
      out so each individual Worker invocation completes within 15 minutes. That
      orchestration is a documented escalation path (see the module-level
      "Step Functions escalation" note below) rather than the default wiring;
      the function's 15-minute ``timeout`` here bounds any single invocation.

Least-privilege IAM (Requirement 6.1):
    The Worker is granted only what it needs:
      * Bedrock — invoke foundation models / inference profiles (LLM-judge
        metric backends in ``uaef[server]``).
      * DynamoDB — read/write the Jobs table (lifecycle state) and the UAEF
        experiment table (``persist=True``).
      * S3 — read/write the payload bucket (request/result payloads) and the
        UAEF results bucket (full result JSON).

Step Functions escalation (Requirement 6.4) — documented, not default:
    For batches that cannot complete in one 15-minute invocation, wrap this
    Worker in a Step Functions state machine:
      1. A splitter step partitions ``traces``/``ground_truths`` into chunks each
         sized to finish well within 15 minutes.
      2. A ``Map`` state invokes :pyattr:`worker_function` once per chunk
         (each invocation independently bounded by this function's 15-minute
         ``timeout``), with Step Functions handling retries and fan-out.
      3. A reducer step records the aggregate Job outcome.
    Step Functions is intentionally NOT provisioned by default (it is overkill
    for single evaluations and small batches — see the design's "Worker
    Invocation Strategy" tradeoff table); it is the escalation path when batch
    size demands it.

This module is part of the standalone ``uaef-service`` deployable app and does
NOT import the ``uaef`` package.

Requirements: 2.3, 6.1, 6.3, 6.4, 6.5
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from aws_cdk import (
    NestedStack,
    Duration,
    Size,
    BundlingOptions,
    aws_lambda as lambda_,
    aws_ecr_assets as ecr_assets,
    aws_iam as iam,
)
from constructs import Construct

from .constants import (
    JOBS_TABLE_NAME,
    PAYLOAD_BUCKET_NAME,
    UAEF_EXPERIMENT_TABLE_NAME,
    UAEF_RESULTS_BUCKET_NAME,
    UAEF_VERSION,
)

# --------------------------------------------------------------------------- #
# Packaging paths and tunables.
# --------------------------------------------------------------------------- #

#: App root (``uaef-service/``) — the bundling asset source. Contains the
#: ``handlers/`` package and the root ``job_state``/``schemas``/``validation``
#: modules the Worker imports. ``infra/`` CDK code is deliberately NOT copied
#: into the Lambda artifact.
_APP_ROOT = Path(__file__).resolve().parent.parent

#: Lambda runtime used for the zip-asset Worker.
_RUNTIME = lambda_.Runtime.PYTHON_3_12

#: Handler entry point: ``handlers/worker.py`` -> ``handler``.
_HANDLER = "handlers.worker.handler"

#: Single-invocation execution limit. The 15-minute Lambda hard cap (Req 6.4):
#: batches expected to exceed this must be escalated to Step Functions.
_TIMEOUT = Duration.minutes(15)

#: Generous memory for the heavy ``uaef[server]`` workload (CPU scales with it).
_MEMORY_MB = 4096

#: Extra ephemeral ``/tmp`` for large payloads streamed through the Worker.
_EPHEMERAL_STORAGE = Size.gibibytes(2)

#: Async retries. The Worker's PENDING->PROCESSING conditional write makes it
#: idempotent, but duplicate processing is wasteful, so we disable automatic
#: async retries and rely on the Job lifecycle guard.
_ASYNC_RETRY_ATTEMPTS = 0

#: CDK context flag selecting the container-image packaging fallback (Req 6.3).
_CONTAINER_IMAGE_CONTEXT = "worker_container_image"

#: CDK context key: comma-separated extra S3 bucket names that
#: ``invoke_evaluate``'s ``s3_data_path`` may reference, on top of the service
#: payload bucket (security review H-03). Set at deploy time with
#: ``-c worker_allowed_data_buckets=my-bucket,another-bucket``. Mirrors the
#: runtime env var read by ``handlers/worker.py::_allowed_data_buckets`` — kept
#: in sync here so the IAM grant and the handler's own check agree.
_ALLOWED_DATA_BUCKETS_CONTEXT = "worker_allowed_data_buckets"

#: Env var name the Worker handler reads the same allow-list from at runtime.
_ALLOWED_DATA_BUCKETS_ENV = "UAEF_ALLOWED_DATA_BUCKETS"

#: Relative path to the container-image fallback Dockerfile (from app root).
_DOCKERFILE = "docker/worker/Dockerfile"

#: Service handler files copied alongside the installed package (zip + image).
_HANDLER_SOURCES = "handlers job_state.py schemas.py validation.py"


class WorkerStack(NestedStack):
    """Worker Lambda packaged with the pinned ``uaef[server]``.

    Attributes:
        worker_function: The Worker Lambda (zip-asset by default, container
            image when ``-c worker_container_image=true``). The API Lambda
            async-invokes this function; grant it invoke rights via
            :meth:`grant_async_invoke`.
        uses_container_image: ``True`` when the container-image fallback
            (Requirement 6.3) was selected for packaging.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        storage: Optional[Construct] = None,
        uaef_dependency: str,
        **kwargs,
    ) -> None:
        """Create the Worker stack.

        Args:
            storage: The :class:`infra.storage_stack.StorageStack` providing the
                Jobs table, payload bucket, and the referenced UAEF experiment
                table / results bucket the Worker is granted access to. May be
                ``None`` while sibling stacks are still being authored, in which
                case resource grants/env are wired from constant names only.
            uaef_dependency: The exact pinned pip spec the Worker installs —
                ``uaef[server]==X.Y.Z`` from
                :data:`infra.constants.UAEF_PINNED_DEPENDENCY`. Single source of
                truth (Requirement 2.3).
        """
        super().__init__(scope, construct_id, **kwargs)

        # Default to container-image packaging. `pip install uaef` now pulls the
        # full capability set (all metric backends, adapters, analysis, judge,
        # client), which exceeds Lambda's 250 MB unzipped zip limit, so the
        # Worker must ship as a container image (10 GB limit). Operators with a
        # slimmed dependency set can opt back into zip packaging with
        # `-c worker_container_image=false`.
        _ctx = self.node.try_get_context(_CONTAINER_IMAGE_CONTEXT)
        if _ctx is None:
            self.uses_container_image = True
        elif isinstance(_ctx, str):
            self.uses_container_image = _ctx.strip().lower() not in {"false", "0", "no"}
        else:
            self.uses_container_image = bool(_ctx)

        environment = self._build_environment(storage)

        if self.uses_container_image:
            # --- Container-image fallback (Requirement 6.3) ----------------
            # Selected when curated ``uaef[server]`` exceeds the 250 MB unzipped
            # zip limit. Installs the SAME pinned spec via build arg, so the
            # single-source-of-truth contract holds.
            self.worker_function = lambda_.DockerImageFunction(
                self,
                "WorkerFunction",
                code=lambda_.DockerImageCode.from_image_asset(
                    directory=str(_APP_ROOT),
                    file=_DOCKERFILE,
                    build_args={"UAEF_DEPENDENCY": uaef_dependency},
                    platform=ecr_assets.Platform.LINUX_AMD64,
                ),
                timeout=_TIMEOUT,
                memory_size=_MEMORY_MB,
                ephemeral_storage_size=_EPHEMERAL_STORAGE,
                environment=environment,
                retry_attempts=_ASYNC_RETRY_ATTEMPTS,
            )
        else:
            # --- Default zip-asset Lambda (Requirement 6.3) ----------------
            # CDK bundles the asset by ``pip install``-ing the pinned
            # ``uaef[server]`` into the artifact and copying ONLY the service's
            # handler code beside it (no vendored UAEF source — Requirement 2.3).
            self.worker_function = lambda_.Function(
                self,
                "WorkerFunction",
                runtime=_RUNTIME,
                handler=_HANDLER,
                code=lambda_.Code.from_asset(
                    str(_APP_ROOT),
                    bundling=BundlingOptions(
                        image=_RUNTIME.bundling_image,
                        command=[
                            "bash",
                            "-c",
                            # Install the pinned library (preferring a locally
                            # built wheel under ./wheels for the registry-free
                            # Path B; falls back to the configured index), then
                            # copy handler code.
                            f'pip install --no-cache-dir --find-links wheels "{uaef_dependency}" '
                            "-t /asset-output "
                            f"&& cp -r {_HANDLER_SOURCES} /asset-output/",
                        ],
                    ),
                ),
                timeout=_TIMEOUT,
                memory_size=_MEMORY_MB,
                ephemeral_storage_size=_EPHEMERAL_STORAGE,
                environment=environment,
                retry_attempts=_ASYNC_RETRY_ATTEMPTS,
            )

        # Least-privilege IAM for Bedrock / DynamoDB / S3 (Requirement 6.1).
        self._grant_bedrock()
        self._grant_agent_invocation()
        self._grant_persistence(storage)
        self._grant_experiment_provisioning()

        # Provisioned concurrency: keep 1 warm instance to avoid cold-start
        # timeouts on synchronous actions (validate-data, catalog, experiments,
        # reports, compare). Container-image Lambdas have longer cold starts
        # (~10-30s) which can exceed the API Gateway 30s integration cap.
        _provisioned = self.node.try_get_context("worker_provisioned_concurrency")
        provisioned_count = 0
        if _provisioned is not None:
            try:
                provisioned_count = int(_provisioned)
            except (ValueError, TypeError):
                provisioned_count = 0

        if provisioned_count > 0:
            version = self.worker_function.current_version
            alias = lambda_.Alias(
                self,
                "WorkerLiveAlias",
                alias_name="live",
                version=version,
                provisioned_concurrent_executions=provisioned_count,
            )
            # Expose the alias as the function to invoke (callers use the alias
            # ARN so traffic always hits a warm instance).
            self.worker_function_alias = alias
        else:
            self.worker_function_alias = None

    # ----------------------------------------------------------------- #
    # Environment
    # ----------------------------------------------------------------- #

    def _build_environment(self, storage: Optional[Construct]) -> dict:
        """Build the Worker's environment.

        Includes the payload bucket the handler reads ``requestRef`` from, the
        Jobs table name, the UAEF persistence targets (so ``persist=True`` writes
        to the referenced experiment table / results bucket), and the expected
        pinned version for the worker's version-mismatch guard (Requirement 2.4).
        """
        jobs_table = getattr(storage, "jobs_table", None)
        payload_bucket = getattr(storage, "payload_bucket", None)

        return {
            # Worker handler reads ``requestRef`` payloads and writes results.
            "PAYLOAD_BUCKET_NAME": (
                payload_bucket.bucket_name if payload_bucket is not None
                else PAYLOAD_BUCKET_NAME
            ),
            # Extra buckets invoke_evaluate's s3_data_path may read from,
            # beyond the payload bucket (security review H-03). Empty by
            # default — deployers opt in explicitly. Kept in sync with the IAM
            # grant in ``_grant_agent_invocation``.
            _ALLOWED_DATA_BUCKETS_ENV: ",".join(self._allowed_data_bucket_names()),
            # Jobs table lifecycle state.
            "JOBS_TABLE_NAME": (
                jobs_table.table_name if jobs_table is not None
                else JOBS_TABLE_NAME
            ),
            # UAEF library persistence targets (library defaults; persist=True).
            "UAEF_DYNAMODB_TABLE": UAEF_EXPERIMENT_TABLE_NAME,
            "UAEF_S3_BUCKET": UAEF_RESULTS_BUCKET_NAME,
            # Single-source-of-truth guard: worker compares uaef.__version__.
            "UAEF_EXPECTED_VERSION": UAEF_VERSION,
            # DeepEval writes config to $HOME/.deepeval; Lambda's root fs is
            # read-only so redirect to the writable /tmp.
            "DEEPEVAL_HOME": "/tmp/.deepeval",  # nosec B108 - Lambda runtime path (only /tmp is writable), set at deploy time
            "HOME": "/tmp",  # nosec B108 - Lambda runtime path (only /tmp is writable), set at deploy time
        }

    # ----------------------------------------------------------------- #
    # IAM (least privilege — Requirement 6.1)
    # ----------------------------------------------------------------- #

    def _grant_bedrock(self) -> None:
        """Allow invoking Bedrock foundation models / inference profiles only."""
        self.worker_function.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/*",
                ],
            )
        )

    def _allowed_data_bucket_names(self) -> list:
        """Extra bucket names ``s3_data_path`` may reference (CDK context)."""
        raw = self.node.try_get_context(_ALLOWED_DATA_BUCKETS_CONTEXT) or ""
        return [b.strip() for b in raw.split(",") if b.strip()]

    def _grant_agent_invocation(self) -> None:
        """Allow the Worker to invoke reachable agents for invoke-and-evaluate.

        Supports the deployed-UI "server invokes the agent" path: the Worker
        drives an AgentCore runtime or a Bedrock agent to collect a trace per
        question, then evaluates them. HTTP agents need no IAM (plain outbound
        HTTPS). Local agents are unreachable from Lambda and use the notebook
        path instead.

        Least-privilege (Requirement 6.1 / security review H-04): scoped to
        AgentCore runtimes and Bedrock agent aliases in THIS account only. The
        caller-supplied ``agent_runtime_arn`` / ``agent_id`` + ``alias_id``
        (request body ``connection`` fields) select *which* same-account
        resource to invoke; they can no longer reach a runtime or agent in a
        different account, which the previous ``Resource: "*"`` grant allowed
        (a request-driven confused-deputy / cross-account proxy risk).

        The read-only CloudWatch Logs grant that used to accompany this method
        has been removed: the deployed Worker always invokes
        ``AgentCoreAdapter.invoke(..., fetch_logs=False)`` (see
        ``handlers/worker.py::_invoke_agent``) because polling OTEL logs after
        each invocation can take up to ~3 minutes per row and would blow the
        Lambda timeout on a multi-row batch. No code path on the deployed
        service reads CloudWatch Logs, so the grant was pure excess privilege
        (any log group in the account was readable).

        This method also grants read access to any CDK-context allow-listed
        data buckets (``worker_allowed_data_buckets``) for invoke-evaluate's
        ``s3_data_path`` — see the S3 statement below and
        ``handlers/worker.py::_read_data_bytes``. The previous
        ``s3:GetObject`` on ``arn:aws:s3:::*/*`` has been removed; it let any
        caller's ``s3_data_path`` read any object in any bucket the Worker
        role could reach (security review H-03).
        """
        self.worker_function.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["bedrock-agentcore:InvokeAgentRuntime"],
                resources=[
                    f"arn:aws:bedrock-agentcore:*:{self.account}:runtime/*",
                ],
            )
        )
        self.worker_function.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["bedrock:InvokeAgent"],
                resources=[
                    # InvokeAgent is authorized against the agent-alias
                    # resource, not the agent resource (AWS-documented format:
                    # arn:aws:bedrock:<region>:<account>:agent-alias/<agentId>/<aliasId>).
                    f"arn:aws:bedrock:*:{self.account}:agent-alias/*",
                ],
            )
        )

        # Read ground-truth CSV/XLSX files for invoke-evaluate's s3_data_path
        # (security review H-03). Scoped to the payload bucket plus any
        # explicitly allow-listed buckets — NOT ``arn:aws:s3:::*/*``. The
        # payload bucket is already covered by ``_grant_persistence`` below
        # (``grant_read_write``); this statement only needs to add the extra
        # allow-listed buckets. If none are configured, this statement grants
        # nothing extra (no resources to add), matching the handler-side check
        # in ``handlers/worker.py::_read_data_bytes``.
        extra_buckets = self._allowed_data_bucket_names()
        if extra_buckets:
            self.worker_function.add_to_role_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["s3:GetObject"],
                    resources=[
                        f"arn:aws:s3:::{name}/*" for name in extra_buckets
                    ],
                )
            )

    def _grant_experiment_provisioning(self) -> None:
        """Allow the library to auto-provision its experiment store on first persist.

        ``uaef.api`` initializes the store via ``create_table_if_not_exists()`` /
        ``create_bucket_if_not_exists()``. The data-access grants in
        :meth:`_grant_persistence` don't include creation, so add the (scoped)
        create/describe permissions for the referenced experiment table and
        results bucket. These are no-ops when the resources already exist.
        """
        self.worker_function.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["dynamodb:CreateTable", "dynamodb:DescribeTable"],
                resources=[
                    f"arn:aws:dynamodb:*:{self.account}:table/{UAEF_EXPERIMENT_TABLE_NAME}"
                ],
            )
        )
        self.worker_function.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                # CreateBucket to provision; ListBucket so head_bucket can probe
                # existence. Object RW is granted in _grant_persistence.
                actions=["s3:CreateBucket", "s3:ListBucket"],
                resources=[f"arn:aws:s3:::{UAEF_RESULTS_BUCKET_NAME}"],
            )
        )

    def _grant_persistence(self, storage: Optional[Construct]) -> None:
        """Grant scoped DynamoDB and S3 access to the storage resources.

        Grants are applied only for handles the storage stack exposes (so this
        stack stays robust while siblings are authored): the Jobs table and the
        referenced UAEF experiment table (read/write), and the payload bucket
        plus the referenced UAEF results bucket (read/write).
        """
        if storage is None:
            return

        fn = self.worker_function

        jobs_table = getattr(storage, "jobs_table", None)
        if jobs_table is not None:
            jobs_table.grant_read_write_data(fn)

        uaef_experiment_table = getattr(storage, "uaef_experiment_table", None)
        if uaef_experiment_table is not None:
            uaef_experiment_table.grant_read_write_data(fn)

        payload_bucket = getattr(storage, "payload_bucket", None)
        if payload_bucket is not None:
            payload_bucket.grant_read_write(fn)

        uaef_results_bucket = getattr(storage, "uaef_results_bucket", None)
        if uaef_results_bucket is not None:
            uaef_results_bucket.grant_read_write(fn)

    # ----------------------------------------------------------------- #
    # Async-invoke wiring (Requirement 6.5)
    # ----------------------------------------------------------------- #

    def grant_async_invoke(self, grantee: iam.IGrantable) -> iam.Grant:
        """Grant ``lambda:InvokeFunction`` so the API Lambda can Event-invoke.

        The API Lambda async-invokes the Worker with ``InvocationType='Event'``
        (direct invoke — Requirement 6.5). ``grant_invoke`` covers both sync and
        async invocation. The API stack calls this once it is wired in.
        """
        return self.worker_function.grant_invoke(grantee)
