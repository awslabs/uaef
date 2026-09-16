# Implementation Plan: UAEF Dual Delivery

## Overview

This plan turns the UAEF Dual Delivery design into incremental, test-driven coding steps.
Work proceeds from the single source of truth outward: first correct the library packaging
and dependency split, add the canonical metric-catalog entry point and the publishing
pipeline, refactor the demo backend to remove duplicated logic, ship the optional client
SDK, then build the separate `uaef-service` application (schemas, job state machine, API
Lambda, Worker Lambda, CDK stacks) that consumes the published package without importing
its source.

Language: Python (library, SDK, and service handlers). CDK is authored in Python.
Property-based tests use `hypothesis`; AWS-facing tests use `moto` / DynamoDB Local.

Conventions:
- The library lives in `src/uaef/` (this repo). The service is a separate deployable
  app under `uaef-service/` that never imports UAEF source — it `pip install`s the package.
- Test sub-tasks are marked `*` and are optional.

## Tasks

- [x] 1. Correct library packaging and dependency split
  - [x] 1.1 Rewrite `pyproject.toml` into lean core plus opt-in extras
    - Set core `dependencies` to exactly `{pydantic, boto3, pyyaml, jsonpath-ng, numpy}`
    - Define extras exactly: `integrations`, `langgraph`, `analysis`, `judge`, `client`, `server`, `notebooks`, `dev`, `all`; `server = ["uaef[integrations]", "uaef[analysis]"]`; `all` aggregates the opt-in sets
    - Ensure no extra dependency is duplicated back into core; add `build`/`twine` to `dev`
    - Bump `version` to `0.2.0`
    - _Requirements: 1 (intro), 3.1, 3.2, 3.3_
  - [ ]* 1.2 Write unit test asserting the dependency manifest is correct
    - Parse `pyproject.toml`; assert core dependency set equality, exact extra names, and that no extra package name appears in core
    - _Requirements: 3.1, 3.2, 3.3_
  - [x] 1.3 Add ImportError guards to integration modules
    - Guard `ragas`, `deepeval`, `langchain-community`, `datasets` imports in `src/uaef/integrations/*` and the matching `src/uaef/metrics/{ragas_metrics,deepeval_metrics}.py`
    - Raise an `ImportError` whose message names the missing extra and the exact `pip install 'uaef[<extra>]'` command, and return no metric result
    - _Requirements: 3.5_
  - [ ]* 1.4 Write property test for core-only import and guard messages
    - **Property 2: Extras are real and additive**
    - **Validates: Requirements 3.4, 3.5**
  - [ ]* 1.5 Write packaging-size test
    - Build the `uaef[server]` set and assert unzipped size is under 250MB (or that the container-image fallback is selected)
    - _Requirements: 6.3_

- [x] 2. Add the canonical full metric-catalog entry point
  - [x] 2.1 Implement `get_full_metric_catalog()`
    - In `src/uaef/metrics/__init__.py` (re-exported from `uaef.api`), return every metric name grouped by dimension/integration by merging the built-in metric registry with `uaef.integrations.registry`
    - Omit any integration group whose extra is not installed, without raising
    - _Requirements: 12.6, 12.8_
  - [ ]* 2.2 Write unit test for catalog merging and omission
    - Assert built-ins plus installed integration groups are present; simulate a missing extra and assert its group is omitted with no error
    - _Requirements: 12.6, 12.8_

- [x] 3. Build the publishing pipeline
  - [x] 3.1 Add CI publish stage on version tags
    - In `gitlab-ci.yml`, add a stage gated on tags matching `^v\d+\.\d+\.\d+$` that runs `python -m build` (wheel + sdist), `twine check dist/*`, and `twine upload` to PyPI / CodeArtifact via CI secrets
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  - [x] 3.2 Add version-conflict and build-failure handling
    - Detect an already-published version and fail the publish without overwriting, reporting the conflicting version; halt the release on build failure with a build-failure error
    - _Requirements: 1.5, 1.6_
  - [x] 3.3 Replace install docs with registry-only instructions
    - Update `README.md`: remove clone-the-repo and local-wheel instructions; document `pip install uaef` (PyPI) and CodeArtifact index configuration plus extras
    - _Requirements: 1.7_

- [x] 4. Refactor the demo backend to remove duplicated logic
  - [x] 4.4 Promote agent-invocation glue into the library
    - Move `_invoke_bedrock_agent`, `_invoke_http_agent`, `_invoke_langfuse_trace`, `_invoke_http_agent_for_framework` from `demo/backend/main.py` into a library invocation helper (e.g. `src/uaef/adapters/invocation.py`) alongside the adapters; export it
    - _Requirements: 12.1_
  - [x] 4.1 Replace hand-rolled comparison with `ComparisonEngine`/`compare_runs`
    - Rewrite the `compare_experiments` route in `demo/backend/main.py` to call `uaef.api.compare_runs` / `ComparisonEngine`; on library comparison failure return an error indication (no presentation-layer fallback)
    - _Requirements: 12.2, 12.3, 12.9_
  - [x] 4.2 Read stored aggregates instead of recomputing
    - In `run_evaluation` / `get_experiment_results`, read stored `average_scores` and `overall_average_score`; if absent, return an error indication rather than recomputing
    - _Requirements: 12.4, 12.5_
  - [x] 4.3 Source metric lists from the catalog function
    - Replace the hardcoded RAGAS/DeepEval/Stickler lists in `get_metrics` with a call to `get_full_metric_catalog()`
    - _Requirements: 12.7_
  - [ ]* 4.5 Write no-duplication guard test
    - **Property 11: No business-logic duplication**
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.7**
    - Assert the demo backend imports `ComparisonEngine`/`compare_runs`, reads stored aggregates, and sources metric names from `get_full_metric_catalog`; assert no hand-rolled comparison/aggregation/hardcoded metric lists remain

- [x] 5. Ship the optional UAEFClient SDK
  - [x] 5.1 Create the client package skeleton with submit and poll
    - Add `src/uaef/client/__init__.py` with `UAEFClient(endpoint, token, poll_interval=2.0, timeout=900.0*... )` honoring a 2s poll interval and 15-minute timeout; implement `_submit` and `_poll` (poll until terminal, raise on `FAILED` with the job's error code/message)
    - _Requirements: 10.1, 10.4, 10.5, 10.6, 10.8_
  - [x] 5.2 Implement `evaluate` and `batch_evaluate`
    - Mirror the core arguments of `uaef.api.evaluate` / `batch_evaluate`; POST then poll and return the result object
    - _Requirements: 10.2, 10.3, 10.4_
  - [x] 5.3 Add presigned-upload offload for large payloads
    - When a payload exceeds the 10MB API Gateway body limit, upload via the `/payloads` presigned path before submitting and reference it by key
    - _Requirements: 10.7_
  - [x]* 5.4 Write property test for SDK argument parity
    - **Property 9: SDK parity**
    - **Validates: Requirements 10.2, 10.3**
  - [x]* 5.5 Write unit tests for the poll loop
    - Mock HTTP; test 2s polling, 15-minute timeout error, and `FAILED` raising with error code/message
    - _Requirements: 10.4, 10.5, 10.6, 10.8_

- [x] 6. Checkpoint — library, catalog, SDK, and publishing
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Scaffold the uaef-service app: schemas and storage
  - [x] 7.1 Define request/response schemas
    - Create `uaef-service/schemas.py` with `EvaluateRequest`, `BatchEvaluateRequest`, `JobCreatedResponse`, `JobStatusResponse`, `PresignResponse` per the Low-Level Design; exclude custom-metric / GenericJSON schema-mapping fields
    - _Requirements: 4.1, 4.2, 4.6, 11.2_
  - [x] 7.2 Define the Storage CDK stack
    - Create `uaef-service/infra/storage_stack.py` with the `uaef-service-jobs` DynamoDB table (PK `jobId`, TTL on `ttl`) and the payload S3 bucket; reference the UAEF experiment table/bucket
    - _Requirements: 4.1, 7.1_

- [x] 8. Implement the job state machine
  - [x] 8.1 Implement state transitions with conditional writes
    - Create `uaef-service/job_state.py` with `_transition(job_id, frm, to, **attrs)` doing a conditional DynamoDB update (only when stored status equals `frm`); enforce `PENDING → PROCESSING → COMPLETED|FAILED`, never skipping `PROCESSING`, and reject writes to terminal jobs
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  - [x]* 8.2 Write property test for job lifecycle monotonicity
    - **Property 3: Job lifecycle monotonicity**
    - **Validates: Requirements 5.1, 5.2, 5.6**
    - Generate random transition sequences and assert status only advances forward and terminal states are never overwritten
  - [x]* 8.3 Write unit tests for transition guards
    - Test rejecting non-`PENDING` pickup, terminal immutability on retry/duplicate invocation, and the 1024-char error-message bound
    - _Requirements: 5.3, 5.5, 5.6_

- [x] 9. Implement the API Lambda (does NOT import uaef)
  - [x] 9.1 Implement the router handler
    - Create `uaef-service/handlers/api.py` `handler(event, context)` dispatching by `(httpMethod, resource)` for `/evaluate`, `/batch-evaluate`, `/jobs/{jobId}`, `/experiments`, `/experiments/{id}`, `/payloads`, `/metrics`
    - _Requirements: 6.2_
  - [x] 9.2 Implement job creation with async worker invoke
    - Create `uaef-service/handlers/jobs.py` `create_evaluate_job(event, operation)`: validate body (400 on invalid, no job), write `PENDING`, record caller `sub` as `createdBy`, async-invoke the Worker (`InvocationType='Event'`), return `jobId` within 3s; on enqueue failure set `FAILED` and return an error, performing no synchronous evaluation
    - _Requirements: 4.1, 4.2, 4.3, 4.5, 4.6, 9.3_
  - [x] 9.3 Implement job status with per-caller authorization
    - In `uaef-service/handlers/jobs.py` `get_job_status(job_id, caller_sub)`: 404 when missing; 403 when `createdBy != sub` (omitting status/result); include result ref/summary on `COMPLETED` and error code/message on `FAILED`; add presigned GET for `resultRef`
    - _Requirements: 4.4, 4.7, 9.4_
  - [x] 9.4 Implement presigned payload upload
    - Create `uaef-service/handlers/payloads.py` `presign_payload_upload(caller_sub)`: return a presigned S3 PUT URL + unique key expiring within 15 minutes scoped to one key; on failure return an error and no key; return 413 with guidance when an oversized payload is sent inline; reject requests referencing a missing/expired key without creating a job
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [x] 9.5 Implement the metric-catalog endpoint
    - Create `uaef-service/handlers/catalog.py` `get_metric_catalog()` delegating to the library `get_full_metric_catalog()`; no hardcoded metric names
    - _Requirements: 11.3, 12.7_
  - [x] 9.6 Implement curated-metric validation and limitation enforcement
    - Create `uaef-service/validation.py` accepting only Curated_Metric_Set names; reject custom metrics (400, explains limitation, references library mode, no job, no worker invoke) and `GenericJSONAdapter` schema mappings
    - _Requirements: 11.1, 11.2, 11.3_
  - [x]* 9.7 Write unit tests for the API Lambda
    - Test routing, 400/404/403/413 cases, async-invoke path, `createdBy` recording, and limitation rejections; assert the API Lambda never imports `uaef`
    - _Requirements: 4.3, 4.6, 4.7, 6.2, 7.3, 9.4, 11.1, 11.2_

- [x] 10. Implement the Worker Lambda (imports the published package)
  - [x] 10.1 Implement the worker handler and evaluation path
    - Create `uaef-service/handlers/worker.py` `handler(event, context)`: transition `PENDING → PROCESSING` (idempotent), load request inline or from S3 `requestRef`, call `uaef.api.evaluate`/`batch_evaluate` with `persist` as requested, write `COMPLETED` (+ `experimentId`, `resultRef`/`resultSummary`) or `FAILED` (+ error); add no scoring/aggregation logic
    - _Requirements: 5.2, 5.4, 5.5, 6.1, 8.1, 8.2, 8.3, 8.5_
  - [x] 10.2 Add version-mismatch guard
    - In `worker.py`, compare the loaded `uaef.__version__` against the pinned exact version; on mismatch fail the operation, advance the job only to `FAILED`, and report expected vs loaded versions
    - _Requirements: 2.3, 2.4_
  - [x] 10.3 Implement result storage with persistence fidelity
    - Create `uaef-service/handlers/result_store.py` `_store_result(job_id, result)` writing full result JSON to S3 and returning `resultRef`; ensure `persist=True` goes through UAEF persistence (no partial rows/JSON on failure) and `persist=False`/omitted writes nothing
    - _Requirements: 8.2, 8.3, 8.4_
  - [ ]* 10.4 Write tests for result equivalence and persistence fidelity
    - **Property 5: Result equivalence**
    - **Property 6: Persistence fidelity**
    - **Validates: Requirements 8.1, 8.2, 8.5**

- [x] 11. Checkpoint — service handlers
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Author the CDK nested stacks and wire the service
  - [x] 12.1 Create the parent app and Auth (Cognito) stack
    - Add `uaef-service/app.py` (parent `UaefServiceStack`) and `uaef-service/infra/auth_stack.py` with a Cognito User Pool + app client; pin `uaef[server]==0.2.0` as the worker dependency (single source of truth, no vendored source)
    - _Requirements: 2.1, 2.2, 9.1_
  - [x] 12.2 Create the API stack
    - Add `uaef-service/infra/api_stack.py`: API Gateway REST, Cognito authorizer on protected routes (401 on missing/expired/invalid JWT, no API Lambda invoke), routes, throttling, access logs, API Lambda wiring; API Lambda excludes `uaef`/server extras
    - _Requirements: 6.2, 9.1, 9.2_
  - [x] 12.3 Create the Worker stack
    - Add `uaef-service/infra/worker_stack.py`: Worker Lambda packaged with `uaef[server]` (container-image fallback when over 250MB), async-invoke permissions, least-privilege IAM for Bedrock/DynamoDB/S3; document Step Functions escalation for batches exceeding 15 minutes
    - _Requirements: 2.3, 6.1, 6.3, 6.4, 6.5_
  - [x] 12.4 Create the optional UI stack
    - Add `uaef-service/infra/ui_stack.py` for Amplify hosting of the evolved demo UI (optional, behind a feature flag)
    - _Requirements: 12.1_
  - [ ]* 12.5 Write infrastructure synthesis tests
    - Synthesize the stacks and assert the jobs table TTL, payload bucket, Cognito authorizer on protected routes, and pinned `uaef[server]==0.2.0` worker dependency; assert the dependency-resolution failure path is reported
    - _Requirements: 2.1, 2.5, 4.1, 7.4, 9.1_

- [ ] 13. End-to-end integration tests
  - [ ]* 13.1 Write end-to-end async-lifecycle integration test
    - **Property 1: Single source of truth**
    - **Property 4: Async correctness**
    - **Property 8: Payload bound respected**
    - **Validates: Requirements 2.2, 2.3, 4.1, 4.2, 4.3, 7.2**
    - Install `uaef[server]` into a clean env, run the Worker against DynamoDB Local + S3 (moto): POST → poll → `COMPLETED` with persisted experiment; assert no UAEF source exists in the service tree and oversized payloads route through S3
  - [ ]* 13.2 Write authorization integration test
    - **Property 7: Authorization**
    - **Property 10: Documented limitation holds**
    - **Validates: Requirements 9.1, 9.2, 9.4, 11.1, 11.2, 11.4, 11.5**
    - Assert protected routes require a valid JWT, callers read only their own jobs, custom-metric/GenericJSON requests are rejected server-side, and both remain available in library mode

- [x] 14. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP.
- Each task references the specific requirement clauses it implements for traceability.
- Property-based tests (Properties 3 and 9) use `hypothesis`; the remaining properties are
  validated through unit, integration, packaging-size, and no-duplication guard tests.
- The no-duplication guard test (Task 4.5, Property 11) and the job-state-machine
  monotonicity property test (Task 8.2, Property 3, Requirement 5) are explicit deliverables.
- Checkpoints (Tasks 6, 11, 14) provide incremental validation points.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "4.4", "7.1", "8.1", "12.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "2.2", "3.1", "5.1", "7.2", "8.2", "8.3", "9.1", "9.2", "9.4", "9.5", "9.6", "10.1", "12.2", "12.3"] },
    { "id": 2, "tasks": ["1.4", "3.2", "3.3", "4.1", "5.2", "9.3", "10.2", "10.3", "12.4"] },
    { "id": 3, "tasks": ["1.5", "4.2", "5.3", "5.4", "9.7", "10.4", "12.5"] },
    { "id": 4, "tasks": ["4.3", "5.5"] },
    { "id": 5, "tasks": ["4.5", "13.1", "13.2"] }
  ]
}
```
