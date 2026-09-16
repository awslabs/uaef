# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""ApiStack — API Gateway REST + Cognito authorizer + API Lambda.

This nested stack provisions the public HTTP surface of the UAEF Service. It is
part of the standalone ``uaef-service`` deployable app and, like its siblings,
does NOT import the ``uaef`` package.

What it provisions
------------------
* **API Lambda** (``handlers/api.py`` → ``api.handler``). This is the request
  router only: it validates input, creates Job records, async-invokes the
  Worker Lambda, and serves status/results. Per **Requirement 6.2** it MUST NOT
  bundle ``uaef`` or the ``server`` extra — the asset is packaged with only the
  lightweight runtime deps (``boto3`` is provided by the Lambda runtime;
  ``pydantic`` is installed into the bundle) and the heavy ``worker.py`` handler
  (the only module that ``import uaef``) is excluded from the asset.
* **API Gateway REST API** with a **Cognito authorizer**
  (:class:`~aws_cdk.aws_apigateway.CognitoUserPoolsAuthorizer`) bound to the
  service User Pool. Every protected route carries
  ``authorization_type=COGNITO`` so API Gateway validates the JWT's signature,
  ``exp`` claim, and issuer *before* the integration runs. A request that
  presents no JWT, an expired JWT, or one failing signature/issuer validation is
  rejected with a 401 and the API Lambda is never invoked
  (**Requirements 9.1, 9.2**).
* **Routes** mapping onto UAEF's ``evaluate`` / ``batch_evaluate`` / experiments
  API (see design "API Route Definitions").
* **Throttling** and **access logs** configured on the deployment stage.

Wiring
------
The parent (:class:`UaefServiceStack`) constructs this stack with the Cognito
``user_pool`` / ``user_pool_client`` and the :class:`StorageStack` handles. The
API Lambda is granted read/write on the jobs table and read/write (presign) on
the payload bucket. The Worker function name is injected later (task 12.3) via
:meth:`bind_worker` once the Worker stack exists; until then a placeholder env
var is set so the stack synthesizes standalone.

Requirements: 6.2, 9.1, 9.2
"""

from __future__ import annotations

import os
from typing import Optional

from aws_cdk import (
    Annotations,
    NestedStack,
    Duration,
    RemovalPolicy,
    BundlingOptions,
    aws_apigateway as apigateway,
    aws_cognito as cognito,
    aws_lambda as _lambda,
    aws_logs as logs,
    aws_iam as iam,
    aws_wafv2 as wafv2,
)
from constructs import Construct

from .constants import (
    APP_NAME,
    JOBS_TABLE_NAME,
    PAYLOAD_BUCKET_NAME,
)
from .cors import resolve_cors_origins

# Path to the Lambda handler source tree (``uaef-service/handlers``), resolved
# relative to this module so the asset works regardless of the synth CWD.
_HANDLERS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, "handlers")
)

# Files excluded from the API Lambda asset. ``worker.py`` is the only handler
# that ``import uaef`` — keeping it out of the API bundle enforces Requirement
# 6.2 (the API Lambda must not bundle UAEF or the server extra). Caches and
# compiled artifacts are excluded for a stable asset hash.
_API_ASSET_EXCLUDE = ["worker.py", "__pycache__", "*.pyc", "*.pyo"]

# The API Lambda imports root-level modules (``job_state``/``schemas``/
# ``validation``) in addition to the ``handlers`` package, so the asset is
# bundled from the app root (``uaef-service/``) — mirroring the Worker. Only the
# files the API Lambda needs are copied into the artifact (see the bundling
# command); ``worker.py`` is dropped because it is the sole ``uaef`` importer
# (Requirement 6.2). Everything below is excluded from the asset input so the
# hash stays small/stable and infra/library/build dirs are never bundled.
_APP_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
_ROOT_ASSET_EXCLUDE = [
    ".venv", "cdk.out", "infra", "docker", "ui", "tests", "wheels",
    ".pytest_cache", ".hypothesis", "node_modules", "__pycache__",
    "*.pyc", "*.pyo",
]

# Lightweight deps the API Lambda needs beyond the runtime-provided ``boto3``.
# Deliberately NOT ``uaef`` / ``uaef[server]`` (Requirement 6.2).
_API_LAMBDA_DEPENDENCIES = "pydantic>=2.0"

# Placeholder Worker function name set until the Worker stack binds the real one
# via :meth:`ApiStack.bind_worker` (task 12.3).
_WORKER_FUNCTION_PLACEHOLDER = "UNBOUND"


class ApiStack(NestedStack):
    """API Gateway REST API, Cognito authorizer, and the routing API Lambda.

    Attributes:
        api: The REST API fronting the service. Protected routes require a valid
            Cognito JWT (the authorizer rejects invalid/missing tokens with 401
            before the API Lambda runs).
        api_lambda: The router Lambda (``api.handler``). Packaged without the
            ``uaef`` library or the ``server`` extra (Requirement 6.2).
        authorizer: The Cognito authorizer bound to the service User Pool.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        user_pool: cognito.IUserPool,
        user_pool_client: Optional[cognito.IUserPoolClient] = None,
        storage: Optional[Construct] = None,
        cors_origin: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self._user_pool = user_pool
        self._user_pool_client = user_pool_client
        self._storage = storage

        # --- CORS allowed origins (security review M-01) -------------------
        # Resolved by the shared helper so this API and the payload bucket
        # (StorageStack) always allow the exact same origins.
        #
        # with_ui: the parent passes ``cors_origin`` — the deployed UI's own
        # CloudFront origin as a CloudFormation token — so the real origin is
        # known within a single deploy and NO wildcard is needed. This removes
        # the old chicken-and-egg (UiStack creates the distribution but is built
        # after ApiStack) that used to force a wildcard "bootstrap" deploy
        # followed by a redeploy with the real origin.
        #
        # No derived origin (with_eks / UI-less API-only): the origin(s) must be
        # supplied via the ``api_cors_allowed_origins`` context, and a wildcard
        # is rejected unless ``i_acknowledge_insecure_cors=true``. See
        # ``infra/cors.py`` for the full rules.
        _origins = resolve_cors_origins(self, cors_origin)
        if "*" in _origins:
            Annotations.of(self).add_warning(
                "api_cors_allowed_origins='*' is in effect (i_acknowledge_insecure_cors=true). "
                "This allows any web origin to call this API. Supply the real origin instead "
                "as soon as it is known — see uaef-service/README.md's CORS section."
            )
            self._cors_allowed_origins = apigateway.Cors.ALL_ORIGINS
        else:
            self._cors_allowed_origins = _origins

        # Resolve storage resource names; fall back to the shared constants so
        # the stack synthesizes even if StorageStack is not yet wired.
        jobs_table_name = (
            storage.jobs_table.table_name if storage is not None else JOBS_TABLE_NAME
        )
        payload_bucket_name = (
            storage.payload_bucket.bucket_name
            if storage is not None
            else PAYLOAD_BUCKET_NAME
        )

        # --- API Lambda (router; no uaef / server extra) ------------------
        # boto3 is provided by the Lambda runtime; only pydantic is installed
        # into the asset. ``worker.py`` (the sole uaef importer) is excluded.
        self.api_lambda = _lambda.Function(
            self,
            "ApiLambda",
            function_name=f"{APP_NAME}-api",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="handlers.api.handler",
            code=_lambda.Code.from_asset(
                _APP_ROOT_DIR,
                exclude=_ROOT_ASSET_EXCLUDE,
                bundling=BundlingOptions(
                    image=_lambda.Runtime.PYTHON_3_11.bundling_image,
                    command=[
                        "bash",
                        "-c",
                        # Install only the lightweight deps (NOT uaef), pinned to
                        # the Lambda's x86_64 / cp311 target so the compiled
                        # pydantic-core wheel matches the function regardless of
                        # the build host's architecture (avoids the arm64-on-x86
                        # "No module named 'pydantic_core._pydantic_core'" failure).
                        "pip install --no-cache-dir --only-binary=:all: "
                        "--platform manylinux2014_x86_64 --implementation cp "
                        "--python-version 3.11 "
                        f"--target /asset-output {_API_LAMBDA_DEPENDENCIES} "
                        # Copy the API Lambda's code: the handlers package plus
                        # the root modules it imports. Drop worker.py — the only
                        # handler that imports uaef (Requirement 6.2).
                        "&& cp -r handlers job_state.py schemas.py validation.py /asset-output/ "
                        "&& rm -f /asset-output/handlers/worker.py "
                        "&& rm -rf /asset-output/handlers/__pycache__",
                    ],
                ),
            ),
            timeout=Duration.seconds(29),  # aligned with API Gateway integration cap
            memory_size=256,
            environment={
                # Wired to the real Worker function name later via bind_worker().
                "WORKER_FUNCTION_NAME": _WORKER_FUNCTION_PLACEHOLDER,
                "JOBS_TABLE_NAME": jobs_table_name,
                "PAYLOAD_BUCKET_NAME": payload_bucket_name,
            },
        )

        # --- Least-privilege grants to service storage --------------------
        if storage is not None:
            # Jobs table: API Lambda writes PENDING records and reads status.
            storage.jobs_table.grant_read_write_data(self.api_lambda)
            # Payload bucket: API Lambda presigns PUT/GET URLs (needs object RW).
            storage.payload_bucket.grant_read_write(self.api_lambda)

        # --- AgentCore runtime discovery (GET /agentcore/runtimes) --------
        # The API Lambda lists deployed AgentCore runtimes so the UI can offer a
        # picker. Read-only control-plane access; the Worker (not the API Lambda)
        # holds the invoke permission.
        self.api_lambda.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["bedrock-agentcore:ListAgentRuntimes"],
                resources=["*"],
            )
        )

        # --- Access logs --------------------------------------------------
        access_log_group = logs.LogGroup(
            self,
            "ApiAccessLogs",
            log_group_name=f"/{APP_NAME}/api-access-logs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # --- REST API + stage (throttling + access logs) ------------------
        self.api = apigateway.RestApi(
            self,
            "RestApi",
            rest_api_name=f"{APP_NAME}-api",
            description="UAEF Service HTTP API (Cognito-protected).",
            # Browser SPA (the deployed UI) calls this API cross-origin, so
            # enable CORS preflight on every resource. Preflight OPTIONS are
            # unauthenticated; the actual methods stay Cognito-protected.
            # Origin allow-list resolved above (defaults to "*" unchanged;
            # security review M-01).
            default_cors_preflight_options=apigateway.CorsOptions(
                allow_origins=self._cors_allowed_origins,
                allow_methods=apigateway.Cors.ALL_METHODS,
                allow_headers=["Authorization", "Content-Type"],
            ),
            # API Gateway hard-caps request bodies at 10MB; large payloads use
            # the presigned-S3 path (POST /payloads). See Requirement 7.
            deploy_options=apigateway.StageOptions(
                stage_name="prod",
                throttling_rate_limit=50,
                throttling_burst_limit=100,
                access_log_destination=apigateway.LogGroupLogDestination(
                    access_log_group
                ),
                access_log_format=apigateway.AccessLogFormat.json_with_standard_fields(
                    caller=True,
                    http_method=True,
                    ip=True,
                    protocol=True,
                    request_time=True,
                    resource_path=True,
                    response_length=True,
                    status=True,
                    user=True,
                ),
                logging_level=apigateway.MethodLoggingLevel.INFO,
                metrics_enabled=True,
            ),
        )

        # --- WAFv2 WebACL (security review M-01) --------------------------
        # AWS-managed common rule set (generic OWASP-style protections) plus a
        # rate-based rule as a coarse backstop in front of Cognito's own
        # throttling and the stage's own throttling_rate_limit/burst_limit
        # above. Purely additive — a new layer in front of the existing
        # Cognito-authorizer + throttling posture, not a change to it. Regular
        # (non-CloudFront) WAFv2 scope is REGIONAL, matching this REST API.
        self.web_acl = wafv2.CfnWebACL(
            self,
            "ApiWebAcl",
            default_action=wafv2.CfnWebACL.DefaultActionProperty(allow={}),
            scope="REGIONAL",
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True,
                metric_name=f"{APP_NAME}-api-waf",
                sampled_requests_enabled=True,
            ),
            rules=[
                wafv2.CfnWebACL.RuleProperty(
                    name="AWSManagedRulesCommonRuleSet",
                    priority=0,
                    override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                            vendor_name="AWS",
                            name="AWSManagedRulesCommonRuleSet",
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name=f"{APP_NAME}-api-waf-common",
                        sampled_requests_enabled=True,
                    ),
                ),
                wafv2.CfnWebACL.RuleProperty(
                    name="RateLimit",
                    priority=1,
                    action=wafv2.CfnWebACL.RuleActionProperty(block={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        rate_based_statement=wafv2.CfnWebACL.RateBasedStatementProperty(
                            # Per 5-minute window, per source IP. Generous
                            # relative to the stage's 50 rps / 100 burst
                            # throttle (which already bounds sustained load);
                            # this is a backstop against a single IP flooding
                            # the endpoint rather than the primary control.
                            limit=3000,
                            aggregate_key_type="IP",
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name=f"{APP_NAME}-api-waf-ratelimit",
                        sampled_requests_enabled=True,
                    ),
                ),
            ],
        )
        wafv2.CfnWebACLAssociation(
            self,
            "ApiWebAclAssociation",
            resource_arn=self.api.deployment_stage.stage_arn,
            web_acl_arn=self.web_acl.attr_arn,
        )

        # --- CORS on API Gateway's own error responses --------------------
        # The Cognito authorizer rejects bad/expired tokens with 401 and the
        # gateway emits other 4xx/5xx itself — these bypass the Lambda, so they
        # carry no CORS headers by default and a browser sees an opaque CORS
        # failure instead of the real status. Add Access-Control-* to the
        # gateway responses so the SPA can read them (e.g. to detect a 401 and
        # re-authenticate).
        #
        # Gateway responses take a single static header value (no per-request
        # origin reflection like the Lambda-backed routes get from CorsOptions
        # above), so with a restricted origin list this uses the first
        # configured origin. With multiple allowed origins, only the first is
        # granted read access to gateway-level error responses specifically
        # (security review L-05); the actual routes remain correctly
        # restricted to the full list via default_cors_preflight_options.
        if self._cors_allowed_origins == apigateway.Cors.ALL_ORIGINS:
            _gateway_response_origin = "*"
        else:
            _gateway_response_origin = self._cors_allowed_origins[0]
        _cors_headers = {
            "Access-Control-Allow-Origin": f"'{_gateway_response_origin}'",
            "Access-Control-Allow-Headers": "'Authorization,Content-Type'",
            "Access-Control-Allow-Methods": "'GET,POST,OPTIONS'",
        }
        for _id, _gr in (
            ("Unauthorized", apigateway.ResponseType.UNAUTHORIZED),      # 401
            ("AccessDenied", apigateway.ResponseType.ACCESS_DENIED),     # 403
            ("Default4xx", apigateway.ResponseType.DEFAULT_4_XX),
            ("Default5xx", apigateway.ResponseType.DEFAULT_5_XX),
        ):
            self.api.add_gateway_response(
                f"GatewayResponse{_id}",
                type=_gr,
                response_headers=_cors_headers,
            )

        # --- Cognito authorizer (rejects invalid/missing JWT with 401) ----
        # Bound to the service User Pool. API Gateway validates signature, exp,
        # and issuer at the authorizer BEFORE invoking the integration, so a
        # missing/expired/invalid token yields 401 and the API Lambda is never
        # invoked (Requirements 9.1, 9.2).
        self.authorizer = apigateway.CognitoUserPoolsAuthorizer(
            self,
            "CognitoAuthorizer",
            cognito_user_pools=[user_pool],
            authorizer_name=f"{APP_NAME}-cognito-authorizer",
        )

        # All protected routes share this Cognito authorization.
        self._protected_method_options = apigateway.MethodOptions(
            authorization_type=apigateway.AuthorizationType.COGNITO,
            authorizer=self.authorizer,
        )

        # --- Routes -------------------------------------------------------
        self._add_routes()

        # --- Force a fresh stage deployment when the authorizer pool changes -
        # CDK computes the auto-generated Deployment's logical id from the API
        # MODEL (resources/methods) only — NOT from the Cognito authorizer's
        # providerARNs. So when the target User Pool changes (e.g. auth.mode
        # existing<->new, or a pool recreation), the authorizer resource updates
        # but the *deployed* `prod` stage keeps validating the OLD pool, and
        # every request 401s until a new deployment is created. That surfaced as
        # a Hosted-UI "login loop" (the SPA treats 401 as session-expired and
        # re-logs in forever). Salting the deployment's logical id with the pool
        # identifier changes it whenever the pool changes, so `cdk deploy`
        # creates a new deployment automatically — no manual
        # `aws apigateway create-deployment` needed. See README "Redeploying".
        if self.api.latest_deployment is not None:
            self.api.latest_deployment.add_to_logical_id(
                {"cognitoAuthorizerPool": user_pool.user_pool_id}
            )

    # ------------------------------------------------------------------ #
    # Route wiring
    # ------------------------------------------------------------------ #

    def _add_routes(self) -> None:
        """Attach the service routes, each behind the Cognito authorizer.

        Routes (see design "API Route Definitions"):
            POST   /evaluate
            POST   /batch-evaluate
            GET    /jobs/{jobId}
            GET    /experiments
            GET    /experiments/{experimentId}
            POST   /payloads
            GET    /metrics
        """
        integration = apigateway.LambdaIntegration(self.api_lambda, proxy=True)
        root = self.api.root

        # POST /evaluate
        evaluate = root.add_resource("evaluate")
        evaluate.add_method("POST", integration, **self._method_kwargs())

        # POST /batch-evaluate
        batch_evaluate = root.add_resource("batch-evaluate")
        batch_evaluate.add_method("POST", integration, **self._method_kwargs())

        # GET /jobs/{jobId}
        jobs = root.add_resource("jobs")
        job_by_id = jobs.add_resource("{jobId}")
        job_by_id.add_method("GET", integration, **self._method_kwargs())

        # GET /experiments  and  GET /experiments/{experimentId}
        experiments = root.add_resource("experiments")
        experiments.add_method("GET", integration, **self._method_kwargs())
        experiment_by_id = experiments.add_resource("{experimentId}")
        experiment_by_id.add_method("GET", integration, **self._method_kwargs())

        # POST /payloads (presigned upload URL)
        payloads = root.add_resource("payloads")
        payloads.add_method("POST", integration, **self._method_kwargs())

        # GET /metrics (curated catalog)
        metrics = root.add_resource("metrics")
        metrics.add_method("GET", integration, **self._method_kwargs())

        # GET /agent-types (static connection-field config for the UI)
        agent_types = root.add_resource("agent-types")
        agent_types.add_method("GET", integration, **self._method_kwargs())

        # GET /agentcore/runtimes (discover deployed AgentCore runtimes)
        agentcore = root.add_resource("agentcore")
        runtimes = agentcore.add_resource("runtimes")
        runtimes.add_method("GET", integration, **self._method_kwargs())

        # POST /invoke-evaluate (server invokes a reachable agent, then evaluates)
        invoke_evaluate = root.add_resource("invoke-evaluate")
        invoke_evaluate.add_method("POST", integration, **self._method_kwargs())

        # POST /validate-data (server-side ground-truth parse + preview via uaef.data)
        validate_data = root.add_resource("validate-data")
        validate_data.add_method("POST", integration, **self._method_kwargs())

        # POST /reports (generate a library report for an experiment via uaef.reporting)
        reports = root.add_resource("reports")
        reports.add_method("POST", integration, **self._method_kwargs())

        # POST /compare-experiments (compare multiple experiments side by side)
        compare_experiments = root.add_resource("compare-experiments")
        compare_experiments.add_method("POST", integration, **self._method_kwargs())

    def _method_kwargs(self) -> dict:
        """Common keyword args applying the Cognito authorizer to a method."""
        return {
            "authorization_type": apigateway.AuthorizationType.COGNITO,
            "authorizer": self.authorizer,
        }

    # ------------------------------------------------------------------ #
    # Cross-stack wiring (called by the Worker stack, task 12.3)
    # ------------------------------------------------------------------ #

    def bind_worker(self, worker_function: _lambda.IFunction) -> None:
        """Wire the API Lambda to the Worker Lambda.

        Sets ``WORKER_FUNCTION_NAME`` on the API Lambda and grants it permission
        to asynchronously invoke the Worker (``InvocationType='Event'``). Called
        by the Worker stack once the Worker function exists.
        """
        self.api_lambda.add_environment(
            "WORKER_FUNCTION_NAME", worker_function.function_name
        )
        worker_function.grant_invoke(self.api_lambda)

    def bind_orchestrator(self, state_machine_arn: str) -> None:
        """Wire the API Lambda to the Step Functions evaluation orchestrator.

        Sets ``EVAL_STATE_MACHINE_ARN`` on the API Lambda and grants it
        permission to start executions on the state machine. When this env var
        is set, the API Lambda routes evaluations through Step Functions
        (parallel-by-dimension) instead of directly invoking the Worker.
        """
        self.api_lambda.add_environment("EVAL_STATE_MACHINE_ARN", state_machine_arn)
        self.api_lambda.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["states:StartExecution"],
                resources=[state_machine_arn],
            )
        )

    @property
    def api_url(self) -> str:
        """The invoke URL of the deployed API stage."""
        return self.api.url
