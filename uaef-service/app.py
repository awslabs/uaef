#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CDK entry point for the standalone ``uaef-service`` deployable app.

This file defines the parent :class:`UaefServiceStack`, which orchestrates the
service's nested stacks and wires shared configuration between them (mirroring
the reference app's nested-stack structure described in the design's Deployment
Topology):

    UaefServiceStack (parent)
      ├── AuthStack      — Cognito User Pool + app client            [task 12.1]
      ├── StorageStack   — Jobs DynamoDB table (+TTL) + payload S3   [task 7.2]
      ├── ApiStack       — API Gateway REST + Cognito authorizer     [task 12.2]
      ├── WorkerStack    — Worker Lambda (uaef[server])              [task 12.3]
      └── UiStack        — optional Amplify UI hosting (flagged)     [task 12.4]

Single Source of Truth (Requirements 2.1, 2.2):
    The service consumes the published ``uaef`` package pinned to the exact
    version in :mod:`infra.constants` (``UAEF_PINNED_DEPENDENCY``).
    This app contains NO vendored or re-implemented UAEF source.

The nested stacks beyond AuthStack are authored in subsequent tasks; the parent
instantiates each one as it becomes available and passes the shared handles
(User Pool, jobs table, payload bucket) through. Their wiring is guarded so this
entry point synthesizes cleanly while the remaining stacks are still in flight.
"""

from __future__ import annotations

import aws_cdk as cdk
from constructs import Construct

from infra.auth_stack import AuthStack, ExistingAuthConfig
from infra.constants import APP_NAME, UAEF_PINNED_DEPENDENCY, _DEPLOY_SUFFIX

# Deployment-tracking token embedded in the root stack's CloudFormation
# Description. The AWS solution-tracking tooling reads the deployed stack's
# Description and matches "(uksb-09bv4bh0dn)" to record stack deployments; it
# has no runtime effect. Applied as the parent stack's default description
# below so every deploy (with_ui and with_eks alike) carries it.
_SOLUTION_DESCRIPTION = (
    "UAEF Service: standalone deployment of the Universal Agentic Evaluation "
    "Framework (uksb-09bv4bh0dn)."
)


class UaefServiceStack(cdk.Stack):
    """Parent stack that orchestrates the UAEF Service nested stacks.

    Attributes:
        auth: The :class:`AuthStack` providing the Cognito User Pool + client
            used by the API Gateway authorizer.
    """

    #: The pinned UAEF library dependency the Worker Lambda installs. Exposed on
    #: the parent so nested stacks (and synthesis tests) share one source.
    uaef_dependency: str = UAEF_PINNED_DEPENDENCY

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        # Default the CloudFormation Description to the solution-tracking token
        # (uksb-09bv4bh0dn) unless a caller overrides it. setdefault so tests or
        # alternate entry points can still pass their own description.
        kwargs.setdefault("description", _SOLUTION_DESCRIPTION)
        super().__init__(scope, construct_id, **kwargs)

        # --- Identity (Cognito) -------------------------------------------
        self.auth = AuthStack(self, "Auth")

        # --- Remaining nested stacks (added in later tasks) ----------------
        # StorageStack (task 7.2), ApiStack (task 12.2), WorkerStack (task
        # 12.3), and UiStack (task 12.4) are wired here once authored. The
        # WorkerStack consumes ``self.uaef_dependency`` so the pinned
        # ``uaef[server]==0.2.0`` spec stays the single source of truth.
        self._wire_remaining_stacks()
        self._add_outputs()

    def _add_outputs(self) -> None:
        """Surface the key endpoints/IDs as CloudFormation outputs.

        Printed at the end of ``cdk deploy`` so callers can find the API URL and
        the Cognito IDs needed to obtain a JWT, without digging through the
        console or extra CLI queries.
        """
        cdk.CfnOutput(
            self,
            "ApiBaseUrl",
            value=self.api.api_url,
            description="Base invoke URL of the UAEF Service API (stage 'prod')",
        ) if getattr(self, "api", None) is not None else None
        cdk.CfnOutput(
            self,
            "AuthMode",
            value=self.auth.auth_mode,
            description="Auth mode: 'new' (UAEF-managed pool) or 'existing' (external pool)",
        )
        cdk.CfnOutput(
            self,
            "UserPoolId",
            value=self.auth.user_pool_id,
            description="Cognito User Pool ID (created or imported)",
        )
        cdk.CfnOutput(
            self,
            "AppClientId",
            value=self.auth.user_pool_client_id,
            description="Cognito App Client ID",
        )
        if getattr(self, "worker", None) is not None:
            cdk.CfnOutput(
                self,
                "WorkerFunctionName",
                value=self.worker.worker_function.function_name,
                description="Worker Lambda function name",
            )
        if getattr(self, "orchestrator", None) is not None:
            cdk.CfnOutput(
                self,
                "StateMachineArn",
                value=self.orchestrator.state_machine_arn,
                description="Step Functions state machine ARN for parallel-by-dimension evaluation",
            )
        if getattr(self, "ui", None) is not None:
            cdk.CfnOutput(
                self,
                "UiUrl",
                value=self.ui.ui_url,
                description="Deployed UI URL (CloudFront)",
            )

    def _wire_remaining_stacks(self) -> None:
        """Instantiate and wire nested stacks that exist yet beyond Auth.

        Imports are performed lazily and guarded so the parent synthesizes even
        while later-task stacks are still being authored. As each module lands,
        it is picked up automatically without editing this method's contract.
        """
        # UiStack — optional static SPA on S3 + CloudFront, behind deploy_ui.
        #
        # Built FIRST (it depends only on Auth) so its CloudFront origin is
        # available as a CloudFormation reference to the CORS rules in
        # StorageStack and ApiStack below. That is what lets the whole service
        # — including a real, non-wildcard CORS origin — deploy in a single
        # pass. The UiStack no longer receives the ApiStack: the API URL it
        # needs is written into config.json afterwards by ConfigStack (created
        # last, see below), which keeps the dependency graph acyclic.
        try:
            from infra.ui_stack import UiStack
        except ImportError:
            UiStack = None  # authored in task 12.4

        deploy_ui = self.node.try_get_context("deploy_ui")
        should_deploy_ui = (
            str(deploy_ui).lower() in ("true", "1", "yes")
            if deploy_ui is not None
            else False
        )
        ui = None
        if UiStack is not None and should_deploy_ui:
            ui = UiStack(
                self, "Ui",
                user_pool=self.auth.user_pool,
                auth_mode=self.auth.auth_mode,
            )
        # The deployed UI's CloudFront origin (a CFN token) when a UI is
        # deployed in this run, else None — in which case the CORS origin must
        # be supplied explicitly (see infra/cors.py).
        ui_origin = ui.ui_origin if ui is not None else None

        # StorageStack — Jobs table (+TTL) and payload bucket. The payload
        # bucket's CORS allow-list is locked to the derived UI origin.
        try:
            from infra.storage_stack import StorageStack
        except ImportError:
            StorageStack = None  # authored in task 7.2

        storage = (
            StorageStack(self, "Storage", cors_origin=ui_origin)
            if StorageStack is not None
            else None
        )

        # deploy_compute flag: when False, skip ALL compute stacks (ApiStack +
        # WorkerStack + OrchestratorStack). Used by legacy deploy modes that
        # provision only storage + auth via CDK.
        # deploy_api flag: when False, skip only ApiStack (API Gateway + router
        # Lambda). Worker + Orchestrator are still deployed. Used by the EKS
        # mode which serves the HTTP API itself but delegates evaluation to the
        # Step Functions orchestrator + Worker Lambda.
        deploy_compute = self.node.try_get_context("deploy_compute")
        deploy_api_ctx = self.node.try_get_context("deploy_api")

        # Backward compatibility: deploy_compute=false skips everything.
        skip_compute = str(deploy_compute).lower() == "false" if deploy_compute is not None else False
        # deploy_api defaults to True unless explicitly false.
        skip_api = str(deploy_api_ctx).lower() == "false" if deploy_api_ctx is not None else False

        api = None
        worker = None
        orchestrator = None

        if not skip_compute:
            # ApiStack — API Gateway REST + Cognito authorizer + API Lambda.
            # Skipped in EKS mode (deploy_api=false) where EKS serves the HTTP API.
            if not skip_api:
                try:
                    from infra.api_stack import ApiStack
                except ImportError:
                    ApiStack = None  # authored in task 12.2

                api = (
                    ApiStack(
                        self,
                        "Api",
                        user_pool=self.auth.user_pool,
                        user_pool_client=self.auth.user_pool_client,
                        storage=storage,
                        cors_origin=ui_origin,
                    )
                    if ApiStack is not None
                    else None
                )

            # WorkerStack — Worker Lambda packaged with the pinned uaef[server].
            try:
                from infra.worker_stack import WorkerStack
            except ImportError:
                WorkerStack = None  # authored in task 12.3

            if WorkerStack is not None:
                worker = WorkerStack(
                    self,
                    "Worker",
                    storage=storage,
                    uaef_dependency=self.uaef_dependency,
                )

            # Wire the API Lambda to the Worker: set WORKER_FUNCTION_NAME and grant
            # async invoke, so job creation and the /metrics catalog can reach it.
            if api is not None and worker is not None:
                api.bind_worker(worker.worker_function)

            # OrchestratorStack — Step Functions state machine for parallel-by-
            # dimension evaluation. Routes all evaluations through a Map state that
            # fans out one Worker invocation per metric dimension.
            try:
                from infra.orchestrator_stack import OrchestratorStack
            except ImportError:
                OrchestratorStack = None

            if OrchestratorStack is not None and worker is not None:
                orchestrator = OrchestratorStack(
                    self,
                    "Orchestrator",
                    worker_function=worker.worker_function,
                    storage=storage,
                )

            # Wire the API Lambda to the orchestrator: set EVAL_STATE_MACHINE_ARN
            # so evaluations route through Step Functions instead of direct Worker
            # async-invoke.
            if api is not None and orchestrator is not None:
                api.bind_orchestrator(orchestrator.state_machine_arn)

        # ConfigStack — writes the SPA's runtime config.json into the UI bucket.
        # Created LAST, after both UiStack and ApiStack, because config.json
        # needs the API URL (ApiStack) alongside the UI bucket/client/domain
        # (UiStack). Nothing depends on it, so the graph stays acyclic and a
        # single deploy fills in the real API URL — no second pass. See the
        # UiStack comment above and infra/config_stack.py.
        if ui is not None:
            try:
                from infra.config_stack import ConfigStack
            except ImportError:
                ConfigStack = None

            if ConfigStack is not None:
                ConfigStack(
                    self, "Config",
                    site_bucket=ui.site_bucket,
                    distribution=ui.distribution,
                    user_pool=self.auth.user_pool,
                    ui_client=ui.ui_client,
                    cognito_domain_base_url=ui.cognito_domain_base_url,
                    ui_url=ui.ui_url,
                    api_base_url=api.api_url if api is not None else "",
                )

        # Expose for outputs and tests.
        self.api = api
        self.storage = storage
        self.worker = worker
        self.orchestrator = orchestrator
        self.ui = ui


def main() -> None:
    """Synthesize the CDK app."""
    app = cdk.App()
    UaefServiceStack(app, f"UaefServiceStack-{_DEPLOY_SUFFIX}")
    app.synth()


if __name__ == "__main__":
    main()
