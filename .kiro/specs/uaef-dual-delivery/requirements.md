# Requirements Document

## Introduction

UAEF (the Unified Agentic Evaluation Framework) is consumed today by cloning the
repository or referencing a local wheel file. This feature, **UAEF Dual Delivery**, enables UAEF to be
consumed two ways from a single source of truth without copying or cloning the repository:

1. **Library mode** — the `uaef` package is published to a registry (PyPI for public
   distribution, AWS CodeArtifact for internal) and installed via `pip install uaef`, which
   delivers the full capability set out of the box.
2. **Service mode** — a separately deployable application exposes UAEF over HTTP via
   API Gateway, Cognito, and Lambda. The service is just another consumer of the published
   `uaef` package; it does not vendor, copy, or re-implement UAEF source.

The feature also shapes the dependency structure so that `pip install uaef` installs the
**full capability set by default** (every metric and capability is available out of the box),
retaining the former capability extras only as no-op compatibility aliases; introduces an
async job + polling lifecycle to sidestep the API Gateway 29-second integration timeout;
handles large payloads via presigned S3 references; ships an optional client SDK; documents
the library-only limitation for user-supplied Python; and enforces a strict no-duplication
boundary in which presentation layers contain no evaluation, aggregation, comparison, or
metric-catalog logic of their own.

These requirements are derived from the approved design document and are written so that
each of the design's correctness properties (Properties 1–11) maps to one or more
acceptance criteria.

## Glossary

- **UAEF_Library**: The `uaef` Python package built into a wheel and published to a registry. Owns all evaluation, aggregation, comparison, and metric-catalog logic.
- **Publishing_Pipeline**: The CI stage that builds the wheel and source distribution and publishes them to PyPI and/or AWS CodeArtifact.
- **Registry**: A Python package index — PyPI (public) or AWS CodeArtifact (internal).
- **UAEF_Service**: The separately deployable application (CDK infrastructure plus Lambda handlers) that exposes UAEF over HTTP. Consumes UAEF_Library from the Registry; never imports UAEF source.
- **API_Lambda**: The UAEF_Service Lambda that validates requests, creates job records, asynchronously invokes the Worker_Lambda, and serves status and results. Does not import UAEF_Library.
- **Worker_Lambda**: The UAEF_Service Lambda that imports the pip-installed UAEF_Library and runs evaluation operations.
- **Jobs_Table**: The DynamoDB table (`uaef-service-jobs`) holding async job lifecycle state.
- **Experiment_Store**: UAEF_Library's existing experiment persistence (DynamoDB experiment rows plus full result JSON in S3).
- **Payload_Store**: The S3 location used for presigned upload and retrieval of large request and result payloads.
- **Cognito_Authorizer**: The API Gateway authorizer that validates Cognito-issued JWTs on protected routes.
- **UAEFClient**: The optional client SDK shipped with UAEF_Library (behind the `client` extra) that mirrors `evaluate()` / `batch_evaluate()` and hides the POST-plus-poll mechanics.
- **Metric_Catalog_Function**: The single library entry point (`get_full_metric_catalog()`) that returns every available metric name grouped by dimension/integration, merging the built-in metric registry with the integrations registry.
- **Demo_Backend**: The existing demo application backend (`demo/backend/main.py`) that is refactored to remove duplicated comparison, aggregation, and hardcoded metric-list logic.
- **Job**: A single async evaluation request tracked in the Jobs_Table, with a `status` of `PENDING`, `PROCESSING`, `COMPLETED`, or `FAILED`.
- **Terminal_State**: A Job status of `COMPLETED` or `FAILED`.
- **Server_Extra**: The `uaef[server]` optional-dependency set the Worker_Lambda installs. Now a no-op alias that resolves to the full-capability core (the core already includes every runtime capability), retained so the pinned `uaef[server]==0.2.0` install command keeps resolving.
- **Curated_Metric_Set**: The trusted, server-side metric names accepted by UAEF_Service (built-ins plus integration metrics, all of which ship in the core), excluding user-supplied custom metrics.

## Requirements

### Requirement 1: Registry-Based Library Distribution

**User Story:** As a developer, I want to install UAEF from a package registry, so that I no longer need to clone the repository or reference a local wheel file.

#### Acceptance Criteria

1. WHEN a version-tagged release is triggered, THE Publishing_Pipeline SHALL build a wheel and a source distribution from the UAEF_Library source.
2. WHEN the build of distributions completes successfully, THE Publishing_Pipeline SHALL publish the built distributions to the configured Registry.
3. WHERE the target Registry is PyPI, THE Publishing_Pipeline SHALL publish the distributions so that `pip install uaef` resolves the package from PyPI.
4. WHERE the target Registry is AWS CodeArtifact, THE Publishing_Pipeline SHALL publish the distributions so that `pip install uaef` resolves the package after CodeArtifact index configuration.
5. IF a release attempts to publish a version that already exists in the Registry, THEN THE Publishing_Pipeline SHALL reject the publish, leave the existing published version unchanged, and report a version-conflict error identifying the conflicting version.
6. IF the build of distributions fails, THEN THE Publishing_Pipeline SHALL halt the release without publishing and report a build-failure error.
7. THE UAEF_Library documentation SHALL present registry installation as the only documented install path, replacing the clone-the-repository and local-wheel instructions.

### Requirement 2: Single Source of Truth

**User Story:** As a maintainer, I want the service to depend on the published library as a normal pinned dependency, so that there is exactly one copy of the evaluation logic.

#### Acceptance Criteria

1. THE UAEF_Service SHALL declare UAEF_Library as a dependency pinned to a single exact version using an equality version specifier resolved from the Registry.
2. THE UAEF_Service source tree SHALL contain no copy, vendored snapshot, or re-implementation of UAEF_Library source, such that no file under the UAEF_Service repository defines or duplicates any module of the `uaef` package.
3. WHEN the Worker_Lambda resolves `import uaef`, THE Worker_Lambda SHALL load UAEF_Library from the pip-installed Registry package at the pinned exact version.
4. IF the UAEF_Library version loaded by the Worker_Lambda does not equal the pinned exact version declared by the UAEF_Service, THEN THE Worker_Lambda SHALL fail the operation, leave any associated Job record advanced only to `FAILED`, and report a version-mismatch error identifying the expected and loaded versions.
5. IF the pinned UAEF_Library version cannot be resolved from the Registry during installation, THEN THE UAEF_Service SHALL fail installation and report a dependency-resolution error identifying the unresolved version.

### Requirement 3: Full Capability Core

**User Story:** As a user, I want `pip install uaef` to give me every metric and capability out of the box, so that I can test all metrics without hunting for the right extra to install.

#### Acceptance Criteria

1. THE UAEF_Library SHALL declare the required core dependencies as the full capability set {pydantic, boto3, pyyaml, jsonpath-ng, numpy, deepeval, ragas, langchain-community, datasets, langchain-core, langchain-aws, langgraph, pandas, scikit-learn, openpyxl, requests}, so that a base `pip install uaef` makes every built-in and integration metric importable and runnable without installing an extra.
2. THE UAEF_Library SHALL retain the capability extras named exactly integrations, langgraph, analysis, judge, client, and server as no-op aliases that resolve to the base package, preserving existing install commands and the pinned `uaef[server]==0.2.0`.
3. THE UAEF_Library SHALL declare exactly two real opt-in extras, notebooks and dev, providing the Jupyter kernel and developer tooling respectively, and SHALL retain the `all` extra for compatibility.
4. WHEN UAEF_Library is installed via `pip install uaef`, THE UAEF_Library SHALL complete `import uaef` and expose every metric and capability without raising an ImportError for a missing extra.
5. WHERE the inert ImportError guards around integration backends remain in the source for defensive safety, THE UAEF_Library SHALL keep them satisfied because every capability backend ships in the required core.

### Requirement 4: Async Job Submission Within the API Gateway Timeout

**User Story:** As a service client, I want evaluation requests to return immediately with a job identifier, so that long-running evaluations are not blocked by the API Gateway 29-second timeout.

#### Acceptance Criteria

1. WHEN the API_Lambda receives a POST to `/evaluate` with a schema-valid request body, THE API_Lambda SHALL create a Job record with status `PENDING`, asynchronously invoke the Worker_Lambda, and return a response containing a `jobId` within 3 seconds and within the documented 29-second API Gateway integration timeout.
2. WHEN the API_Lambda receives a POST to `/batch-evaluate` with a schema-valid request body, THE API_Lambda SHALL create a Job record with status `PENDING`, asynchronously invoke the Worker_Lambda, and return a response containing a `jobId` within 3 seconds and within the documented 29-second API Gateway integration timeout.
3. THE API_Lambda SHALL NOT perform synchronous evaluation on the request path.
4. WHEN a client polls GET `/jobs/{jobId}` for an existing Job, THE API_Lambda SHALL return the current Job status; SHALL include the result reference and summary when the status is `COMPLETED`; and SHALL include the error code and message when the status is `FAILED`.
5. IF the asynchronous invocation of the Worker_Lambda fails to enqueue, THEN THE API_Lambda SHALL set the Job status to `FAILED`, perform no evaluation on the request path, and return an error response.
6. IF the API_Lambda receives a POST to `/evaluate` or `/batch-evaluate` with a malformed or schema-invalid body, THEN THE API_Lambda SHALL return a 400 response and SHALL NOT create a Job.
7. IF a client polls GET `/jobs/{jobId}` for a Job that does not exist, THEN THE API_Lambda SHALL return a 404 response.

### Requirement 5: Job Lifecycle Monotonicity

**User Story:** As a service operator, I want job state to advance in one direction only, so that completed or failed results are never corrupted by later writes.

#### Acceptance Criteria

1. THE Worker_Lambda SHALL transition a Job status only along the sequence `PENDING` → `PROCESSING` → `COMPLETED` or `PENDING` → `PROCESSING` → `FAILED`, with no backward transitions and without skipping `PROCESSING`.
2. WHEN the Worker_Lambda begins processing a Job, THE Worker_Lambda SHALL transition the Job from `PENDING` to `PROCESSING` using a conditional write applied only if the stored status equals `PENDING`.
3. IF the Worker_Lambda is invoked for a Job whose stored status is not `PENDING`, THEN THE Worker_Lambda SHALL leave the Job record unchanged and perform no evaluation.
4. WHEN the Worker_Lambda completes evaluation successfully, THE Worker_Lambda SHALL write status `COMPLETED` together with the experiment identifier and result reference or summary, applied only if the stored status equals `PROCESSING`.
5. IF evaluation fails with an exception, timeout, or validation error, THEN THE Worker_Lambda SHALL write status `FAILED` together with an error code and an error message of at most 1024 characters, applied only if the stored status equals `PROCESSING`.
6. WHILE a Job is in a Terminal_State, THE Worker_Lambda SHALL reject subsequent writes and leave the Job record unchanged on any retry or duplicate invocation.

### Requirement 6: Worker Invocation Strategy

**User Story:** As a service operator, I want a dedicated worker for compute with documented fallbacks, so that heavy dependencies are isolated and large workloads remain supportable.

#### Acceptance Criteria

1. THE UAEF_Service SHALL use a dedicated Worker_Lambda as the default compute path for evaluation operations.
2. THE API_Lambda SHALL NOT import or bundle UAEF_Library or the Server_Extra, and SHALL delegate all evaluation operations to the Worker_Lambda.
3. WHERE the full-capability UAEF_Library install exceeds the 250 MB Lambda unzipped zip package limit, THE UAEF_Service SHALL package the Worker_Lambda as a container image not exceeding the 10 GB image size limit by default.
4. WHERE the operator opts out of container-image packaging via the `-c worker_container_image=false` context flag, THE UAEF_Service SHALL package the Worker_Lambda as a zip asset instead.
5. WHERE a batch evaluation's expected execution time exceeds the Lambda 15-minute execution limit, THE UAEF_Service SHALL orchestrate the evaluation via Step Functions such that each individual Worker_Lambda invocation completes within the 15-minute limit.
6. WHERE a batch evaluation's expected execution time does not exceed the Lambda 15-minute execution limit, THE UAEF_Service SHALL run the evaluation by directly invoking the Worker_Lambda.

### Requirement 7: Large Payload Handling via Presigned S3 References

**User Story:** As a service client, I want to submit payloads larger than the API Gateway body limit, so that large batch evaluations are not rejected for size alone.

#### Acceptance Criteria

1. WHEN the API_Lambda receives a POST to `/payloads`, THE API_Lambda SHALL return a presigned S3 upload URL and a unique object key that the client references in a subsequent evaluation request.
2. WHERE a request payload exceeds the API Gateway 10MB body limit and is provided by S3 reference, THE API_Lambda SHALL accept the request and process the referenced payload up to a maximum size of 5GB, rather than rejecting it for size alone.
3. IF a payload exceeding the 10MB body limit is sent inline in the request body, THEN THE API_Lambda SHALL return a 413 response that includes guidance to use the presigned upload path, and SHALL NOT create a Job.
4. THE API_Lambda SHALL issue presigned upload URLs that expire no later than 15 minutes after issuance and are scoped to a single object key for a single upload operation.
5. IF generation of the presigned upload URL fails, THEN THE API_Lambda SHALL return an error response indicating the upload URL could not be issued and SHALL NOT return an object key.
6. IF an evaluation request references an S3 key for which no object has been uploaded or whose presigned URL has expired, THEN THE API_Lambda SHALL reject the request with an error response indicating the referenced payload is unavailable and SHALL NOT create a Job.

### Requirement 8: Result Equivalence and Persistence Fidelity

**User Story:** As a user, I want service-mode results to match library-mode results, so that I can trust the service adds no evaluation logic of its own.

#### Acceptance Criteria

1. WHEN the Worker_Lambda evaluates a trace using the Curated_Metric_Set, THE Worker_Lambda SHALL produce an evaluation result in which every numeric score matches UAEF_Library evaluating the same trace at the same pinned version within an absolute tolerance of 1e-9, and every non-numeric field is identical.
2. WHEN a service request specifies `persist=True`, THE Worker_Lambda SHALL persist the result through UAEF_Library so that every Experiment_Store row field and every S3 result JSON field is identical to the corresponding field produced by library-mode persistence of the same result.
3. WHEN a service request omits `persist` or specifies `persist=False`, THE Worker_Lambda SHALL return the evaluation result without writing any row to the Experiment_Store or any object to the Payload_Store.
4. IF persistence through UAEF_Library fails, THEN THE Worker_Lambda SHALL set the Job status to `FAILED` with an error code and message, and SHALL NOT leave a partial Experiment_Store row or partial S3 result JSON.
5. THE Worker_Lambda SHALL add no evaluation, scoring, or aggregation logic beyond the operations provided by UAEF_Library.

### Requirement 9: Authorization and Per-Caller Job Isolation

**User Story:** As a security stakeholder, I want every protected route to require authentication and callers to access only their own jobs, so that data is isolated per user.

#### Acceptance Criteria

1. THE Cognito_Authorizer SHALL require, on every protected route, a Cognito JWT that has a valid signature, an unexpired `exp` claim, and an issuer matching the configured Cognito user pool.
2. IF a request to a protected route presents no JWT, an expired JWT, or a JWT that fails signature or issuer validation, THEN THE Cognito_Authorizer SHALL reject the request with a 401 response and SHALL NOT invoke the API_Lambda.
3. WHEN a Job is created via POST `/evaluate` or POST `/batch-evaluate`, THE API_Lambda SHALL record the caller's Cognito `sub` claim as the Job's `createdBy` value.
4. IF a caller requests a Job whose `createdBy` value does not equal the caller's Cognito `sub`, THEN THE API_Lambda SHALL return a 403 response and SHALL NOT include the Job's status, result reference, or summary in the response.

### Requirement 10: Optional UAEFClient SDK

**User Story:** As a remote user, I want a client SDK that mirrors the library API, so that I get library-like ergonomics without writing POST-and-poll code.

#### Acceptance Criteria

1. THE UAEFClient SHALL ship with UAEF_Library behind the `client` extra.
2. THE UAEFClient `evaluate` method SHALL accept the same core arguments as `uaef.api.evaluate` and SHALL return a result equal to UAEF_Library's output for the same inputs at the same pinned version.
3. THE UAEFClient `batch_evaluate` method SHALL accept the same core arguments as `uaef.api.batch_evaluate` and SHALL return a result equal to UAEF_Library's output for the same inputs at the same pinned version.
4. WHEN a UAEFClient method is called, THE UAEFClient SHALL submit the request, poll the job status endpoint until a Terminal_State is reached, and return the result, hiding the POST-and-poll mechanics from the caller.
5. WHILE a submitted Job has not reached a Terminal_State, THE UAEFClient SHALL poll the job status endpoint at a fixed interval of 2 seconds.
6. IF a submitted Job does not reach a Terminal_State within 15 minutes, THEN THE UAEFClient SHALL stop polling and raise a timeout error.
7. WHERE a request payload exceeds the API Gateway 10MB body limit, THE UAEFClient SHALL upload the payload via the presigned upload path before submitting the request.
8. IF a polled Job reaches status `FAILED`, THEN THE UAEFClient SHALL raise an error containing the Job's error code and message and SHALL return no result.

### Requirement 11: Documented Library-Only Limitation

**User Story:** As a service user, I want a clear, enforced boundary on user-supplied Python, so that the service stays secure and the limitation is explicit.

#### Acceptance Criteria

1. IF a service request specifies a metric name that is absent from the Curated_Metric_Set (including any custom metric registered via `register_metric`), THEN THE API_Lambda SHALL reject the request with a 400 response whose message explains the limitation and references library mode, SHALL create no Job, and SHALL NOT invoke the Worker_Lambda.
2. IF a service request specifies a `GenericJSONAdapter` schema mapping, THEN THE API_Lambda SHALL reject the request with an error response whose message explains the limitation, SHALL create no Job, and SHALL NOT invoke the Worker_Lambda.
3. WHEN the API_Lambda validates an evaluation request, THE API_Lambda SHALL accept only metric names in the Curated_Metric_Set and reject any others.
4. THE UAEF_Library SHALL continue to support custom metrics registered via `register_metric` in library mode.
5. THE UAEF_Library SHALL continue to support `GenericJSONAdapter` schema mappings in library mode.

### Requirement 12: No Business-Logic Duplication in Presentation Layers

**User Story:** As a maintainer, I want all evaluation, aggregation, comparison, and metric-catalog logic to live only in the library, so that presentation layers never fork that logic into separate copies.

#### Acceptance Criteria

1. THE UAEF_Service SHALL compute evaluation and aggregation results only by calling UAEF_Library functions, and SHALL contain no independent re-implementation of that logic.
2. WHEN the Demo_Backend or UAEF_Service compares experiment runs, THE comparison SHALL be produced by UAEF_Library's `compare_runs` or `ComparisonEngine` entry point rather than a comparison computed in the presentation layer.
3. WHEN the Demo_Backend or UAEF_Service compares experiment runs, THE comparison result SHALL equal the output UAEF_Library produces for the same runs at the same pinned version.
4. WHEN the Demo_Backend or UAEF_Service reports experiment aggregates, THE aggregates SHALL be read from the stored `average_scores` and `overall_average_score` values rather than recomputed in the presentation layer.
5. IF the stored `average_scores` or `overall_average_score` values are absent for a requested experiment, THEN THE Demo_Backend or UAEF_Service SHALL return an error indication rather than recomputing the aggregates in the presentation layer.
6. THE UAEF_Library SHALL expose the Metric_Catalog_Function that returns every available metric name grouped by dimension and integration, merging the built-in metric registry with the integrations registry.
7. WHEN the Demo_Backend or UAEF_Service presents a metric catalog, THE catalog SHALL be sourced from the Metric_Catalog_Function and SHALL contain no hardcoded metric-name list.
8. WHEN the Metric_Catalog_Function is called and an integration extra is not installed, THE Metric_Catalog_Function SHALL omit that integration's metric group from the returned catalog without raising an error.
9. IF UAEF_Library's comparison entry point fails, THEN THE Demo_Backend or UAEF_Service SHALL return an error indication rather than falling back to a comparison computed in the presentation layer.
