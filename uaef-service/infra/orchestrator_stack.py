# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OrchestratorStack — Step Functions state machine for parallel-by-dimension evaluation.

This nested stack provisions a Step Functions state machine that fans out
evaluation work by dimension. Each dimension is evaluated in a separate Worker
Lambda invocation (bounded by the Worker's 15-minute timeout), and a reducer
step merges the per-dimension results into a single Job outcome.

Flow:
    1. The API Lambda starts an execution (instead of directly async-invoking
       the Worker) passing the full evaluation request + jobId.
    2. A "Splitter" step (inline Pass/Lambda) partitions the metric list into
       one item per dimension using the metric catalog.
    3. A Map state invokes the Worker Lambda once per dimension (parallel),
       each with only that dimension's metrics.
    4. A "Reducer" step (Lambda) merges the per-dimension EvaluationResults,
       computes the aggregate overall score, persists the combined experiment,
       and marks the Job COMPLETED (or FAILED on error).

Requirements: 6.4
"""

from __future__ import annotations

import os
from typing import Optional

from aws_cdk import (
    NestedStack,
    Duration,
    BundlingOptions,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_logs as logs,
    RemovalPolicy,
)
from constructs import Construct

from .constants import (
    APP_NAME,
    JOBS_TABLE_NAME,
    PAYLOAD_BUCKET_NAME,
    UAEF_EXPERIMENT_TABLE_NAME,
    UAEF_RESULTS_BUCKET_NAME,
    UAEF_VERSION,
)

_APP_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))


class OrchestratorStack(NestedStack):
    """Step Functions orchestrator for parallel-by-dimension evaluation.

    Attributes:
        state_machine: The Step Functions state machine. The API Lambda starts
            an execution on this machine for every evaluation request.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        worker_function: lambda_.IFunction,
        storage: Optional[Construct] = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self._worker_function = worker_function
        self._storage = storage

        # --- Reducer Lambda -----------------------------------------------
        # Lightweight: only needs boto3 (runtime-provided) + job_state module.
        # Does NOT import uaef.
        self.reducer_function = lambda_.Function(
            self,
            "ReducerFunction",
            function_name=f"{APP_NAME}-reducer",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="handlers.orchestrator.reducer_handler",
            code=lambda_.Code.from_asset(
                _APP_ROOT,
                exclude=[
                    ".venv", "cdk.out", "infra", "docker", "ui", "tests", "wheels",
                    ".pytest_cache", ".hypothesis", "node_modules", "__pycache__",
                    "*.pyc", "*.pyo", "handlers/worker.py",
                ],
                bundling=BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_11.bundling_image,
                    command=[
                        "bash",
                        "-c",
                        "pip install --no-cache-dir --only-binary=:all: "
                        "--platform manylinux2014_x86_64 --implementation cp "
                        "--python-version 3.11 "
                        "--target /asset-output pydantic>=2.0 "
                        "&& cp -r handlers job_state.py schemas.py validation.py /asset-output/ "
                        "&& rm -f /asset-output/handlers/worker.py "
                        "&& rm -rf /asset-output/handlers/__pycache__",
                    ],
                ),
            ),
            timeout=Duration.minutes(5),
            memory_size=512,
            environment=self._build_reducer_environment(storage),
        )

        # Reducer needs read/write on the Jobs table to mark COMPLETED/FAILED,
        # and read/write on the payload bucket to store the merged result.
        # Also needs write access to the experiment table to persist combined results.
        if storage is not None:
            jobs_table = getattr(storage, "jobs_table", None)
            if jobs_table is not None:
                jobs_table.grant_read_write_data(self.reducer_function)
            payload_bucket = getattr(storage, "payload_bucket", None)
            if payload_bucket is not None:
                payload_bucket.grant_read_write(self.reducer_function)
            # Reducer reads experiment data to build aggregate scores.
            uaef_experiment_table = getattr(storage, "uaef_experiment_table", None)
            if uaef_experiment_table is not None:
                uaef_experiment_table.grant_read_write_data(self.reducer_function)
            uaef_results_bucket = getattr(storage, "uaef_results_bucket", None)
            if uaef_results_bucket is not None:
                uaef_results_bucket.grant_read_write(self.reducer_function)

        # --- Splitter Lambda ----------------------------------------------
        # Reads the static catalog.json to split metrics into dimensions.
        self.splitter_function = lambda_.Function(
            self,
            "SplitterFunction",
            function_name=f"{APP_NAME}-splitter",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="handlers.orchestrator.splitter_handler",
            code=lambda_.Code.from_asset(
                _APP_ROOT,
                exclude=[
                    ".venv", "cdk.out", "infra", "docker", "ui", "tests", "wheels",
                    ".pytest_cache", ".hypothesis", "node_modules", "__pycache__",
                    "*.pyc", "*.pyo", "handlers/worker.py",
                ],
                bundling=BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_11.bundling_image,
                    command=[
                        "bash",
                        "-c",
                        "pip install --no-cache-dir --only-binary=:all: "
                        "--platform manylinux2014_x86_64 --implementation cp "
                        "--python-version 3.11 "
                        "--target /asset-output pydantic>=2.0 "
                        "&& cp -r handlers job_state.py schemas.py validation.py /asset-output/ "
                        "&& rm -f /asset-output/handlers/worker.py "
                        "&& rm -rf /asset-output/handlers/__pycache__",
                    ],
                ),
            ),
            timeout=Duration.seconds(30),
            memory_size=256,
            environment={
                "WORKER_FUNCTION_NAME": worker_function.function_name,
                "PAYLOAD_BUCKET_NAME": (
                    storage.payload_bucket.bucket_name
                    if storage is not None
                    else PAYLOAD_BUCKET_NAME
                ),
                "JOBS_TABLE_NAME": (
                    storage.jobs_table.table_name
                    if storage is not None and getattr(storage, "jobs_table", None) is not None
                    else JOBS_TABLE_NAME
                ),
            },
        )

        # Splitter may invoke the Worker synchronously for catalog (fallback).
        worker_function.grant_invoke(self.splitter_function)
        # Splitter reads the payload bucket for requestRef resolution.
        # Splitter also needs read/write on the Jobs table to call start_processing.
        if storage is not None:
            payload_bucket = getattr(storage, "payload_bucket", None)
            if payload_bucket is not None:
                payload_bucket.grant_read(self.splitter_function)
            jobs_table = getattr(storage, "jobs_table", None)
            if jobs_table is not None:
                jobs_table.grant_read_write_data(self.splitter_function)

        # --- Step Functions State Machine ---------------------------------
        self.state_machine = self._build_state_machine()

    def _build_reducer_environment(self, storage: Optional[Construct]) -> dict:
        jobs_table = getattr(storage, "jobs_table", None)
        payload_bucket = getattr(storage, "payload_bucket", None)
        return {
            "JOBS_TABLE_NAME": (
                jobs_table.table_name if jobs_table is not None
                else JOBS_TABLE_NAME
            ),
            "PAYLOAD_BUCKET_NAME": (
                payload_bucket.bucket_name if payload_bucket is not None
                else PAYLOAD_BUCKET_NAME
            ),
            "UAEF_DYNAMODB_TABLE": UAEF_EXPERIMENT_TABLE_NAME,
            "UAEF_S3_BUCKET": UAEF_RESULTS_BUCKET_NAME,
        }

    def _build_state_machine(self) -> sfn.StateMachine:
        """Build the Step Functions state machine for parallel-by-dimension eval."""

        # Step 1: Splitter — partitions metrics into per-dimension work items.
        splitter_step = tasks.LambdaInvoke(
            self,
            "SplitByDimension",
            lambda_function=self.splitter_function,
            output_path="$.Payload",
            retry_on_service_exceptions=True,
        )

        # Step 2: Mark job as PROCESSING.
        # The Worker's handler already does start_processing, but when fanning
        # out we need to do it before the Map (the first Worker that wins the
        # conditional write will process; others will skip). We handle this in
        # the splitter — it calls start_processing as part of its logic.

        # Step 3: Map state — one Worker invocation per dimension.
        worker_task = tasks.LambdaInvoke(
            self,
            "EvaluateDimension",
            lambda_function=self._worker_function,
            retry_on_service_exceptions=True,
            result_path="$.workerResult",
            output_path="$.workerResult.Payload",
        )

        map_state = sfn.Map(
            self,
            "ParallelDimensionMap",
            items_path="$.dimensionItems",
            parameters={
                "jobId.$": "$.jobId",
                "operation.$": "$.operation",
                "request.$": "$$.Map.Item.Value.request",
                "requestRef.$": "$$.Map.Item.Value.requestRef",
                "dimensionName.$": "$$.Map.Item.Value.dimensionName",
                "isPartialDimension": True,
            },
            max_concurrency=10,
            result_path="$.mapResults",
        )
        map_state.iterator(worker_task)

        # Step 4: Reducer — merge per-dimension results and complete the job.
        reducer_step = tasks.LambdaInvoke(
            self,
            "ReduceResults",
            lambda_function=self.reducer_function,
            output_path="$.Payload",
            retry_on_service_exceptions=True,
        )

        # Error handling: if the map or reducer fails, mark the job FAILED.
        fail_job_step = tasks.LambdaInvoke(
            self,
            "FailJob",
            lambda_function=self.reducer_function,
            output_path="$.Payload",
            retry_on_service_exceptions=True,
        )

        # Wire: Splitter -> Map -> Reducer
        # On Map failure, catch and route to FailJob.
        map_state.add_catch(
            fail_job_step,
            errors=["States.ALL"],
            result_path="$.mapError",
        )

        reducer_step.add_catch(
            fail_job_step,
            errors=["States.ALL"],
            result_path="$.reducerError",
        )

        definition = splitter_step.next(map_state).next(reducer_step)

        # Log group for execution history (aids debugging).
        log_group = logs.LogGroup(
            self,
            "OrchLogGroup",
            log_group_name=f"/{APP_NAME}/orchestrator",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        state_machine = sfn.StateMachine(
            self,
            "EvalOrchestrator",
            state_machine_name=f"{APP_NAME}-eval-orchestrator",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=Duration.minutes(60),
            logs=sfn.LogOptions(
                destination=log_group,
                level=sfn.LogLevel.ERROR,
            ),
        )

        return state_machine

    @property
    def state_machine_arn(self) -> str:
        return self.state_machine.state_machine_arn
