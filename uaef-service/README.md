# UAEF Service — Deployment Guide

## Prerequisites

### Tools

- AWS CLI
- Python 3.11+ with `pyyaml` available
- Docker (with buildx for multi-arch images)
- `eksctl` (only if creating a new EKS cluster)
- `helm` (only for `with_eks` mode)
- `kubectl` (only for `with_eks` mode)

### One-time AWS setup

**CDK bootstrap — required, and no script in this repo does it for you.**

```bash
cdk bootstrap aws://<account-id>/<deploy_region>
```

`cdk.json` sets `newStyleStackSynthesis: true`, the Worker is a `DockerImageFunction` built from an image asset (`infra/worker_stack.py`), and the UI uses `BucketDeployment` (`infra/ui_stack.py`) — all three need the bootstrap stack's S3 bucket and ECR repo. Skip this and `cdk deploy` fails inside CodeBuild with a bootstrap error that is easy to miss in the streamed log. Run it once per account + region.

**Bedrock model access.** Enable access to the LLM-judge model (`us.anthropic.claude-sonnet-4-6` by default) in your deploy region. Because that is a `us.` cross-region inference profile, the deployment is limited to `us-east-1`, `us-east-2`, or `us-west-2`.

### Credentials

The principal running `scripts/cloud_deploy.sh` needs:

| Service | Actions |
|---------|---------|
| STS | `GetCallerIdentity` |
| S3 | `CreateBucket`, `ListBucket`, `PutObject` (the deploy source bucket) |
| IAM | `GetRole`, `CreateRole`, `AttachRolePolicy`, `PassRole` |
| CodeBuild | `BatchGetProjects`, `CreateProject`/`UpdateProject`, `StartBuild`, `BatchGetBuilds` |
| CloudWatch Logs | `FilterLogEvents` (log streaming) |
| CloudFormation | `DescribeStacks` (printing outputs) |

No third-party credentials or API keys are needed — Bedrock access is granted via IAM to the Worker Lambda.

> ⚠️ The script creates the `uaef-service-deploy-codebuild-role` service role and attaches the AWS-managed **`AdministratorAccess`** policy to it (`scripts/cloud_deploy.sh`). That role — not your own principal — performs the actual CDK deploy. Use a dev or sandbox account, and scope that policy down before pointing this at anything production.

## Deployment Guide

| Deploy Mode | Type of Deployment | Auth Mode | Cognito Configuration |
|-------------|-----------|--------------|----|
| `with_ui` | CDK with CodeBuild | `new` | Create a Cognito pool |
| `with_ui` | CDK with CodeBuild | `existing` | Use a previously deployed Cognito pool |
| `with_eks` | CDK for storage/auth/orchestrator; EKS for container | `new` | Create a Cognito pool  |
| `with_eks` | CDK for storage/auth/orchestrator; EKS for container | `existing` | Use a previously deployed Cognito pool |

> ⚠️ **Deploy one stack per tenant.** This service is designed to run embedded
> inside a platform that provides user/tenant isolation. Direct deployment in a
> multi-tenant environment without a host platform enforcing tenant isolation
> is **not supported**: a single deployment has one Cognito user pool, one job
> table, one experiment table, and one results bucket shared by every
> authenticated caller. Reads of jobs and experiments are scoped to the
> caller's own Cognito `sub`, but that is defense in depth within a tenant —
> the tenant boundary is the deployment (its own account, or at minimum its own
> stack and `deploy_suffix`). See
> [SECURITY.md](../SECURITY.md#deployment-note-tenant-isolation) for the full
> model and the accepted residual risk.

## Quick Start

### 1. Build UAEF (optional)

`cloud_deploy.sh` rebuilds the wheel from source and copies it into `wheels/` on every run, so you can normally skip this. To build it yourself, run the following inside the `agenticevaluationframework/` directory:

```bash
mkdir -p uaef-service/wheels && uv build && cp dist/uaef-*-py3-none-any.whl uaef-service/wheels/
```

### 2. Create your config file

```bash
cd uaef-service
cp config.yaml.example config.yaml
```

### 3. Edit config.yaml

Open `config.yaml` and set the values for your deployment.

- **`deploy_suffix`** — a unique identifier for this deployment that will be included in all resources' names. Two of the resources it names are S3 buckets, whose names must be unique across **all** of AWS — if the suffix you pick is already taken by someone else's bucket, the deploy fails with an opaque `AWS::EarlyValidation::ResourceExistenceCheck` error. Pick something distinctive.
- **`deploy_mode`** — `"with_ui"` or `"with_eks"`
- **`auth.mode`** — `"new"` or `"existing"`
- **`cors_allowed_origins`** — for `with_ui`, **leave blank**; the deploy script derives the UI's origin automatically. Required only for `with_eks`, or to allow extra origins such as a local dev server. See [CORS Configuration](#cors-configuration).

### 4. Deploy

```bash
./scripts/cloud_deploy.sh
```

One call, including the first deploy. The script reads your config and handles the rest,
CORS included — see [CORS Configuration](#cors-configuration) for how the UI origin is
resolved without a second run.

---

## config.yaml Reference

### Top-Level

| Key | Description |
|-----|-------------|
| `deploy_suffix` | Unique suffix appended to all resource names. Allows multiple deployments in the same account. |
| `deploy_region` | AWS region for all resources (e.g., `"us-east-2"`). Overridable with `AWS_REGION` env var. |
| `deploy_mode` | `"with_ui"` (CDK + CodeBuild) or `"with_eks"` (Docker + Helm) |
| `cors_allowed_origins` | Comma-separated origin(s) allowed to call the API and use presigned payload-bucket URLs (security review M-01). For `with_ui`, **leave blank** — the deploy derives the UI's origin automatically; anything set here is an *additional* origin (e.g. a local dev server). **Required for `with_eks`**, which deploys no UI to derive from. See [CORS Configuration](#cors-configuration) below. |
| `acknowledge_insecure_cors` | Set to `true` only to opt into a wildcard `cors_allowed_origins: "*"` on the `with_eks` / no-UI path. `with_ui` never needs it — the real origin is always derived. See [CORS Configuration](#cors-configuration). |

### `auth` Section

| Key | Create a new Cognito pool | Import existing Cognito pool |
|-----|-------------|---|
| `mode` | `"new"` | `"existing"` |
| `existing_user_pool_id` | leave blank | add Cognito pool ID |
| `existing_user_pool_arn` | leave blank | Cognito pool ARN  |
| `existing_app_client_id` | leave blank | Cognito app client ID  |
| `existing_cognito_domain` | leave blank | Cognito domain URL (optional) |

> **Note:** `mode` is *ownership*, not existence — `"new"` = this stack creates/owns the pool; `"existing"` = import a pool owned elsewhere. If this stack created the pool with `"new"`, keep it on `"new"` across redeploys (don't switch to `"existing"`). Default to `mode: "new"` per stack. See [Redeploying](#redeploying) for why.

### `uaef` Section

| Key | Description |
|-----|-------------|
| `version` | Pinned UAEF library version (e.g., `"0.2.0"`) |
| `extra` | Pip extra to install (e.g., `"server"`) |

### `ui` Section (with_ui mode only)

| Key | Description |
|-----|-------------|
| `deploy` | `true` to deploy the React UI, `false` for API-only |

The UI source is `uaef-service/ui/app`, built by `buildspec.yml` during the CodeBuild deploy and served from CloudFront + S3. It is not configurable — and it is not `demo/frontend`, which is the separate local-only demo app.

### `naming` Section

These prefixes combined with `deploy_suffix` for all AWS resource names.
Changing these prefixes are optional.

### `eks` Section (with_eks mode only)

| Key | Description |
|-----|-------------|
| `cluster_name` | Leave blank to auto-derive from `deploy_suffix` (uaef-{suffix}). Auto-populated after creation. |
| `create_cluster` | If `true` and cluster doesn't exist, creates one with eksctl |
| `node_type` | Instance type for new cluster (e.g., `"t3.medium"`) |
| `node_count` | Number of nodes |
| `namespace` | Kubernetes namespace |
| `replica_count` | Pod replicas |
| `ecr_repo_prefix` | ECR repo name prefix (suffixed with `deploy_suffix`) |
| `helm_release_prefix` | Helm release name prefix (suffixed with `deploy_suffix`) |
| `image_tag` | Docker image tag (default: `"latest"`) |
| `platforms` | Build architectures: `"linux/amd64"`, `"linux/arm64"`, or both comma-separated |
| `resources.cpu_request` | CPU request per pod (e.g., `"500m"`) |
| `resources.memory_request` | Memory request per pod (e.g., `"1Gi"`) |
| `resources.cpu_limit` | CPU limit per pod (e.g., `"2000m"`) |
| `resources.memory_limit` | Memory limit per pod (e.g., `"4Gi"`) |
| `ingress.enabled` | `true` to create an ALB ingress |
| `ingress.class_name` | Ingress class (default: `"alb"`) |
| `ingress.host` | Hostname for ingress |
| `service_account_role_arn` | IRSA role ARN (optional; uses node role if blank) |
| `state_machine_arn` | **Auto-populated** after Phase 1 CDK deploy |
| `worker_function_name` | **Auto-populated** after Phase 1 CDK deploy |
| `ecr_repository` | **Auto-populated** after Phase 2 deploy |
| `service_endpoint` | **Auto-populated** after Phase 2 deploy |

---

## Architecture

### with_ui (CDK — fully serverless)

```
                     ┌─────────────────────────────────────────────┐
                     │         cloud_deploy.sh (with_ui)           │
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
                              CDK via CodeBuild (single phase)
                                            │
                                            ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ AuthStack (Cognito)       │ StorageStack (DDB + S3)                    │
   │ WorkerStack (Lambda)      │ OrchestratorStack (Step Functions)         │
   │ ApiStack (API Gateway)    │ UiStack (CloudFront + S3 static site)     │
   └────────────────────────────────────────────────────────────────────────┘
```

**Runtime architecture:**
```
Browser → CloudFront (React UI) → API Gateway → Lambda (API)
                                                     │
                                                     ▼
                                          Step Function Execution
                                                  │
                                            ┌─────┴─────┐
                                            ▼           ▼
                                      Splitter     (partitions metrics by dimension)
                                            │
                                            ▼
                                      Map State    (parallel Worker Lambda invocations, max concurrency 10)
                                            │
                                            ▼
                                      Reducer      (merges results, computes aggregate score, completes job)
                                            │
                                            ▼
                                      DynamoDB (job COMPLETED) + S3 (full results)
```

All resources provisioned via CDK + CodeBuild. The API Lambda creates jobs and starts Step Functions executions. The orchestrator runs sequentially: Splitter partitions metrics by dimension, the Map state fans out Worker Lambda invocations in parallel (one per dimension), then the Reducer merges results and completes the job.

### with_eks (Two-phase deployment)

```
                     ┌─────────────────────────────────────────────┐
                     │         cloud_deploy.sh (with_eks)          │
                     └──────────────┬──────────────────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────────┐
              ▼                                                 ▼
   Phase 1: CDK via CodeBuild                    Phase 2: Docker + Helm
   (provisions backend resources)                (deploys container to EKS)
              │                                                 │
              ▼                                                 ▼
   ┌───────────────────────────┐                 ┌──────────────────────────┐
   │ AuthStack (Cognito)       │                 │ Build multi-arch image   │
   │ StorageStack (DDB + S3)   │                 │ Push to ECR              │
   │ WorkerStack (Lambda)      │                 │ Helm install to cluster  │
   │ OrchestratorStack (SFN)   │                 │ IAM policy attachment    │
   │ (No API GW, No UI)        │                 └──────────────────────────┘
   └───────────────────────────┘
```

**Phase 1** provisions (same compute as `with_ui`, minus API Gateway and UI):
- Cognito User Pool (or imports existing)
- DynamoDB tables (jobs + experiments)
- S3 buckets (payloads + results)
- Worker Lambda (runs uaef evaluation, 15-min timeout)
- Splitter Lambda (partitions metrics by dimension, 30s timeout)
- Reducer Lambda (merges per-dimension results, 5-min timeout)
- Step Functions state machine (orchestrates parallel-by-dimension evaluation, 60-min timeout)

**Phase 2** deploys:
- Multi-arch Docker image to ECR
- Helm chart to EKS (FastAPI container serving the HTTP API)
- IAM policy granting EKS nodes access to DynamoDB, S3, Bedrock, AgentCore

> Both modes deploy the same orchestration layer (Step Functions + Worker/Splitter/Reducer Lambdas). The difference is that `with_ui` also deploys API Gateway + CloudFront UI, while `with_eks` serves the HTTP API from a container and skips API Gateway.

### Runtime flow (with_eks)

```
Client → EKS Service (FastAPI on port 8080)
              │
              ├── CognitoJWTMiddleware (validates Bearer token against JWKS)
              ├── API routes (same handler code as Lambda version)
              │
              └── POST /evaluate or /invoke-evaluate
                      │
                      ▼
              Step Function Execution
                      │
                ┌─────┴─────┐
                ▼           ▼
          Splitter     (partitions metrics by dimension)
                │
                ▼
          Map State    (parallel Worker Lambda invocations, max concurrency 10)
                │
                ▼
          Reducer      (merges results, computes aggregate score, completes job)
                │
                ▼
          DynamoDB (job COMPLETED) + S3 (full results)
```

When `EVAL_STATE_MACHINE_ARN` is set, the EKS container delegates evaluation to Step Functions → Worker Lambda. The container itself only serves the HTTP API and routes evaluation work to the state machine. Synchronous Worker calls (e.g., `GET /metrics` catalog) still invoke the real Worker Lambda directly.

When `EVAL_STATE_MACHINE_ARN` is **not** set (or `WORKER_MODE=inline`), evaluation runs entirely in-process inside the container via background threads — no external Lambda dependency.

### Step Functions Orchestrator

The orchestrator eliminates the 15-minute single-Lambda timeout constraint by splitting evaluation work across dimensions:

1. **Splitter** — Reads `catalog.json` to partition the requested metrics into per-dimension work items. Initializes progress tracking on the job record.
2. **Map** — Invokes the Worker Lambda once per dimension in parallel (max 10 concurrent). Each Worker evaluates only its assigned metrics against the full trace data.
3. **Reducer** — Merges per-dimension `EvaluationResult` payloads into a single combined outcome. Computes the aggregate overall score, persists the merged experiment to DynamoDB and full results to S3, then marks the job COMPLETED (or FAILED on error).

The state machine has a 60-minute total timeout, allowing multi-dimension evaluations that would exceed the 15-minute single-Lambda limit.

---

## Key Components

### Container (deploy_eks/)

| Path | Role |
|------|------|
| `server/main.py` | FastAPI app: CORS, Cognito JWT middleware, health check |
| `server/auth.py` | Cognito JWT validation (fetches JWKS, verifies signature, extracts `sub` claim) |
| `server/routes.py` | HTTP routes → fake Lambda proxy events → existing handler modules |
| `uaef-service/handlers/` | Same handler modules as the Lambda deployment (synced via `sync-uaef.sh`) |
| `uaef-service/job_state.py` | DynamoDB job lifecycle (PENDING → PROCESSING → COMPLETED/FAILED) |
| `Dockerfile` | Multi-stage build: Python 3.12-slim, layered ML deps, uaef wheel, uvicorn (2 workers) |

### Infrastructure (uaef-service/infra/)

| File | Purpose |
|------|---------|
| `constants.py` | Reads config.yaml and exports naming constants |
| `auth_stack.py` | Cognito User Pool provisioning or import |
| `storage_stack.py` | DynamoDB tables + S3 buckets |
| `worker_stack.py` | Worker Lambda (the only component that imports `uaef`) |
| `orchestrator_stack.py` | Step Functions state machine + Splitter/Reducer Lambdas |
| `api_stack.py` | API Gateway + API Lambda (with_ui mode only) |
| `ui_stack.py` | CloudFront + S3 static UI (with_ui mode only) |

### Handlers (uaef-service/handlers/)

| File | Purpose |
|------|---------|
| `api.py` | Request routing, auth check, dispatches to per-route handlers. Does NOT import uaef. |
| `jobs.py` | Job creation: validates schema, creates PENDING job, async-invokes Worker or Step Functions |
| `worker.py` | Performs synchronous evaluation (the only handler that imports `uaef`) |
| `orchestrator.py` | Splitter (partition metrics by dimension) and Reducer (merge results) |
| `catalog.py` | Metric catalog (reads `catalog.json` or invokes Worker for dynamic catalog) |
| `agents.py` | Agent type listing and AgentCore runtime discovery |
| `payloads.py` | Presigned URL generation for S3 uploads |
| `result_store.py` | Experiment persistence and retrieval |

---

## Deployment Script Details (cloud_deploy.sh)

### with_ui flow

1. Validates wheel exists in `wheels/`
2. Creates S3 source bucket + IAM role for CodeBuild
3. Packages service source → uploads to S3
4. Creates/updates CodeBuild project, starts build with CDK context (`deploy_ui=true`)
5. CDK deploys: Auth + Storage + Worker + Orchestrator (Step Functions) + API Gateway + UI
6. Streams build logs, polls until completion
7. Prints CloudFormation stack outputs (UiUrl, ApiEndpoint, StateMachineArn, UserPoolId, etc.)

### with_eks flow

1. Validates/creates EKS cluster (via eksctl if `create_cluster: true`)
2. **Phase 1**: Same CodeBuild + CDK flow, but with `deploy_ui=false deploy_api=false`
   - Provisions: Storage (DDB + S3) + Auth (Cognito) + Worker Lambda + Orchestrator (Step Functions) — same as `with_ui` minus API Gateway and UI
   - Reads CDK outputs (`StateMachineArn`, `WorkerFunctionName`) → writes to config.yaml
3. **Phase 2**:
   - Runs `generate_eks_values.py` to create `helm/values.yaml` from config.yaml
   - Runs `sync-uaef.sh` to copy handler source into `deploy_eks/uaef-service/`
   - Runs `deploy.sh --push --helm-install` to build multi-arch image, push to ECR, Helm upgrade
4. Auto-populates config.yaml with deployment outputs (ECR repo, service endpoint)

### deploy_eks/deploy.sh flags

| Flag | Action |
|------|--------|
| (none) | Local Docker build only (amd64, for testing) |
| `--push` | Build multi-arch + push to ECR |
| `--push --helm-install` | Build + push + Helm upgrade + IAM policy attachment + pod restart |

---

## Testing the Deployment

### with_ui mode

Open the `UiUrl` from the CloudFormation outputs in your browser. Click "Sign In" to go through the Cognito Hosted-UI flow.

### with_eks mode

Port-forward to reach the service:
```bash
kubectl port-forward svc/uaef-service-<suffix> 8080:8080
```

Get a token (use the region where Cognito was deployed):
```bash
TOKEN=$(aws cognito-idp initiate-auth \
  --region <deploy_region> \
  --client-id <your-app-client-id> \
  --auth-flow USER_PASSWORD_AUTH \
  --auth-parameters USERNAME=<your-username>,PASSWORD=<your-password> \
  --query 'AuthenticationResult.AccessToken' --output text)
```

> **Note:** If you deployed with `auth.mode: "new"`, the pool starts empty. Create a user first:
> ```bash
> aws cognito-idp admin-create-user --region <deploy_region> \
>   --user-pool-id <UserPoolId> --username youruser
> aws cognito-idp admin-set-user-password --region <deploy_region> \
>   --user-pool-id <UserPoolId> --username youruser \
>   --password 'YourPassword123!' --permanent
> ```

Test endpoints:
```bash
# Health (no auth)
curl http://localhost:8080/health

# Metrics catalog (auth required)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/metrics

# Submit evaluation
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"trace":{"input":"test","output":"test","expected_output":"test"},"metrics":["accuracy"],"persist":false}' \
  http://localhost:8080/evaluate

# Poll job status
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/jobs/<jobId>
```

---

## CORS Configuration

Security review M-01: the API Gateway CORS policy and the payload S3 bucket's
CORS rule both need an explicit origin allow-list — there is no wildcard
default. For `with_ui`, `cloud_deploy.sh` supplies that origin for you, so
**leave `cors_allowed_origins` blank** and deploy once.

### How the origin is resolved (with_ui)

The UI's origin is its CloudFront domain, which only exists once the
distribution has been created — so `ApiStack` and `StorageStack` can't be given
it as a fixed string ahead of time. Instead, `UiStack` is created first and
hands its CloudFront domain to the API and payload-bucket CORS rules as a
**CloudFormation reference**, which CloudFormation fills in with the real value
during the deploy. A single deploy therefore locks CORS to the real origin, on
both a first deploy and a redeploy, with **no wildcard at any point**.

This works because the dependency between the UI and the API is one-directional.
The one thing that made it circular — the UI needing the API's URL baked into
its `config.json` — is handled by `ConfigStack`, which is created *last* (after
both the UI and the API exist) and writes `config.json` into the UI bucket. See
`infra/cors.py`, `infra/config_stack.py`, and the wiring in `app.py`.

There is no bootstrap wildcard and no second pass: `cloud_deploy.sh` runs one
CDK deploy inside one CodeBuild run, from one script call.

### Extra origins

Anything you set in `cors_allowed_origins` is treated as an **additional**
origin and merged with the derived one — useful for a local dev server:

```yaml
cors_allowed_origins: "http://localhost:5173"
```

Extras never replace the derived CloudFront origin, and a stale `"*"` left in
config is discarded once the real origin is known.

### with_eks

Required. That mode deploys no CloudFront distribution, so there is nothing to
derive from, and the payload bucket's CORS rules still need an origin:

```yaml
cors_allowed_origins: "https://your-client.example.com"
```

### Symptoms of a wrong or stale origin

If `cors_allowed_origins` doesn't exactly match the domain the browser loaded the
UI from, the deploy **succeeds** and the UI loads, but every API call is blocked
by the browser. What you see:

- The **Agent Framework dropdown is empty and won't open** (the `/agent-types`
  call fails and the UI swallows the error).
- The Experiments tab shows `Could not load experiments: Failed to fetch —
  [API] network/CORS failure: GET https://<api-id>.execute-api.<region>.amazonaws.com/prod/experiments`.
- The browser console shows a CORS error naming a missing
  `Access-Control-Allow-Origin` header.

All of these are the *same* fault — every call is failing, not just experiments.
Fix it by setting `cors_allowed_origins` to the exact `UiUrl` and redeploying;
the UI itself doesn't need rebuilding, only the API's allow-list changes.

Common cause: manually setting `cors_allowed_origins` to a value that doesn't
match the domain the browser actually loads the UI from. For `with_ui` you
normally leave it blank and let the deploy derive the real origin.

## Redeploying

> ⚠️ **Upgrading a config.yaml created before the security review.** `cors_allowed_origins`
> is now required with no default, so a `config.yaml` written before it was introduced does
> not contain the key and the deploy script will stop immediately (before making any change)
> until you add it:
>
> ```yaml
> cors_allowed_origins: "https://<your-UiUrl-host>"
> acknowledge_insecure_cors: false
> ```
>
> Use your existing UI origin — retrieve it with
> `aws cloudformation describe-stacks --stack-name UaefServiceStack-<suffix> --query 'Stacks[0].Outputs' --output table`
> and take `UiUrl`. This also applies to `with_eks`, where the payload bucket's CORS rules
> use it even though no API Gateway is deployed. See [CORS Configuration](#cors-configuration).

Just run `./scripts/cloud_deploy.sh` again. Both modes are idempotent:
- **with_ui**: CodeBuild + CDK only deploys the changeset (unchanged resources are untouched)
- **with_eks**: Phase 1 (CDK) only applies changes to backend resources — unchanged DynamoDB/S3/Cognito are untouched. Phase 2 rebuilds only changed Docker layers, Helm upgrades in-place, pods restart.

> ⚠️ **Keep `deploy_suffix` and `auth.mode` constant across redeploys of the same stack.**
>
> `auth.mode` controls **who owns the Cognito pool in this stack**, not whether a pool exists:
> - `new` = *this* stack creates and owns the pool. Once created it's part of the stack, so redeploying with `new` keeps the **same pool, domain, and users**. **Keep it on `new`.**
> - `existing` = the pool is owned **elsewhere** (another stack or created by hand); this stack only imports a reference to it.
>
> **Do not switch a stack from `new` to `existing` on redeploy just because the pool "already exists"** — that removes the pool/domain from this stack's template. The pool is retained (orphaned) but the stack stops managing the Hosted-UI domain, and a later switch back to `new` provisions a *fresh* pool with a new ID. Symptoms: users "don't exist" after redeploy, or `Client does not exist` on the Hosted UI.
>
> Likewise, **changing `deploy_suffix` creates a brand-new separate stack**, not an update to the existing one.
>
> **Sharing one pool/domain across stacks:** only use `existing` to point at a *long-lived* auth stack that is never recreated. A Cognito Hosted-UI domain can attach to only one pool at a time, so if the owning stack recreates its pool, the domain moves with it and every other stack importing that domain breaks (`Client does not exist`). When in doubt, give each stack its own `auth.mode: new`.

---

## Teardown

> ⚠️ **`cdk destroy` does NOT delete your data.** The DynamoDB tables (`uaef-service-jobs-*`, `uaef-experiments-*`), the S3 buckets (`uaef-service-payloads-*`, `uaef-results-*`), and the Cognito **User Pool** are created with `RemovalPolicy.RETAIN` (see `infra/storage_stack.py` / `auth_stack.py`). On `cdk destroy` they are **orphaned, not deleted** — CloudFormation removes them from the stack but leaves the physical resources (and their data + users) in place. This is intentional so a teardown never destroys saved experiments. You must delete them **manually** if you want them gone (see "Manual cleanup" below).

> **There is no teardown script.** `scripts/` contains only `cloud_deploy.sh`; the steps
> below are manual.

> **Grab the Cognito `UserPoolId` before you delete the stack** if you intend to delete the
> pool too — it's in the stack outputs, and it's more awkward to find afterwards:
> `aws cloudformation describe-stacks --stack-name UaefServiceStack-<suffix> --region <deploy_region> --query 'Stacks[0].Outputs' --output table`

### with_ui
```bash
# Delete the CDK stack (removes API Gateway, Lambdas, Step Functions, UI/CloudFront).
# NOTE: DynamoDB tables, S3 buckets, and the Cognito pool are RETAINED (not deleted).
aws cloudformation delete-stack --stack-name UaefServiceStack-<suffix> --region <deploy_region>

# Watch it complete (reports an error once the stack is fully gone)
aws cloudformation describe-stacks --stack-name UaefServiceStack-<suffix> \
  --region <deploy_region> --query 'Stacks[0].StackStatus'
```

> **Why `delete-stack` and not `cdk destroy`:** `cdk destroy` re-synthesizes the app, which
> re-runs the required-context check and fails with
> `ValueError: api_cors_allowed_origins CDK context is required` before deleting anything.
> `delete-stack` skips synthesis entirely. If you prefer `cdk destroy`, you must feed it a
> throwaway origin: `cdk destroy UaefServiceStack-<suffix> -c api_cors_allowed_origins=https://unused.invalid`

Also clean up the deploy scaffolding, which lives **outside** the stack and is not removed
by deleting it:

```bash
aws codebuild delete-project --name uaef-service-deploy-<suffix> --region <deploy_region>
aws s3 rm s3://uaef-service-deploy-src-<account-id>-<deploy_region>/source-<suffix>.zip --region <deploy_region>
```

Leave the IAM role `uaef-service-deploy-codebuild-role` and the
`uaef-service-deploy-src-*` bucket in place unless nothing else deploys from the account —
both are shared across suffixes, not per-deploy.

### with_eks
```bash
# Remove helm release
helm uninstall uaef-service-<suffix>

# Delete the CDK stack (removes Worker, Orchestrator, Lambdas).
# NOTE: DynamoDB tables, S3 buckets, and the Cognito pool are RETAINED (not deleted).
# Use delete-stack, not `cdk destroy` — see the with_ui note above for why.
aws cloudformation delete-stack --stack-name UaefServiceStack-<suffix> --region <deploy_region>

# Detach and delete the IAM policy created for EKS nodes
POLICY_ARN="arn:aws:iam::<account-id>:policy/uaef-service-eks-policy-<cluster-name>"
aws iam detach-role-policy --role-name <node-role-name> --policy-arn $POLICY_ARN
aws iam delete-policy --policy-arn $POLICY_ARN

# Delete cluster (if you created it)
eksctl delete cluster --name uaef-<suffix> --region <deploy_region>

# Clean up ECR
aws ecr delete-repository --repository-name uaef-service-eks-<suffix> --force --region <deploy_region>
```

### Manual cleanup of retained resources

`cdk destroy` leaves these behind (RETAIN). Delete them by hand only if you truly want the data/users gone — this is irreversible:

```bash
REGION=<deploy_region>; SUFFIX=<suffix>

# DynamoDB tables (job state + experiment metadata)
aws dynamodb delete-table --table-name uaef-service-jobs-$SUFFIX --region $REGION
aws dynamodb delete-table --table-name uaef-experiments-$SUFFIX --region $REGION

# S3 buckets (must be emptied before deletion)
aws s3 rm s3://uaef-service-payloads-$SUFFIX --recursive --region $REGION
aws s3 rb s3://uaef-service-payloads-$SUFFIX --region $REGION
aws s3 rm s3://uaef-results-$SUFFIX --recursive --region $REGION
aws s3 rb s3://uaef-results-$SUFFIX --region $REGION

# Cognito User Pool (deletes all users). Delete its hosted-UI domain first if set.
aws cognito-idp delete-user-pool-domain --domain uaef-service-login-$SUFFIX --user-pool-id <UserPoolId> --region $REGION
aws cognito-idp delete-user-pool --user-pool-id <UserPoolId> --region $REGION
```

> Because these names include `deploy_suffix`, a **retained** bucket/table/domain from a prior deploy can block a fresh deploy that reuses the same suffix (e.g. S3 "BucketAlreadyExists", or a Cognito domain still attached to an old pool). Either clean them up as above, or use a new `deploy_suffix`.

---

## Resources — EKS Deployment

| Resource | Name Pattern | Purpose |
|----------|------|---------|
| EKS Cluster | uaef-{deploy_suffix} | Hosts the UAEF service containers |
| ECR Repository | uaef-service-eks-{deploy_suffix} | Docker image registry |
| S3 Bucket | uaef-service-payloads-{deploy_suffix} | Transient staging: request payloads, per-job result JSON |
| S3 Bucket | uaef-results-{deploy_suffix} | Persistent experiment results (written by uaef when `persist=True`) |
| DynamoDB Table | uaef-service-jobs-{deploy_suffix} | Job state tracking (PENDING → PROCESSING → COMPLETED/FAILED) |
| DynamoDB Table | uaef-experiments-{deploy_suffix} | Experiment metadata and average scores |
| Worker Lambda | uaef-service-{suffix}-worker | Evaluates metrics (imports uaef, 15-min timeout) |
| Splitter Lambda | uaef-service-splitter | Partitions metrics by dimension (30s timeout) |
| Reducer Lambda | uaef-service-reducer | Merges results, completes job (5-min timeout) |
| Step Functions | uaef-service-{suffix}-eval-orchestrator | Parallel-by-dimension evaluation (60-min timeout) |
| IAM Policy | uaef-service-eks-policy-{cluster} | Grants EKS nodes access to DynamoDB, S3, Bedrock, AgentCore |
| Helm Release | uaef-service-{deploy_suffix} | Kubernetes deployment (FastAPI, 2 replicas) |
| Service | ClusterIP:8080 | Internal endpoint (access via `kubectl port-forward`) |
| Cognito User Pool | uaef-service-user-pool-{deploy_suffix} | JWT authentication |

---

## Available API Endpoints

| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| GET | /health | Health check (K8s probes) | No |
| GET | /metrics | List available evaluation metrics | Yes |
| GET | /agent-types | Supported agent connection types | Yes |
| GET | /agentcore/runtimes | List available AgentCore runtimes | Yes |
| POST | /evaluate | Submit a single evaluation job | Yes |
| POST | /batch-evaluate | Submit a batch evaluation job | Yes |
| GET | /jobs/{jobId} | Poll job status/results | Yes |
| GET | /experiments | List past experiments | Yes |
| GET | /experiments/{id} | Get experiment detail | Yes |
| POST | /payloads | Get presigned S3 upload URL | Yes |
| POST | /invoke-evaluate | Invoke agent and evaluate result | Yes |
| POST | /validate-data | Validate ground-truth data format | Yes |
| POST | /reports | Generate evaluation report | Yes |
| POST | /compare-experiments | Compare multiple experiments | Yes |

---

## Workflow

1. User logs in to UI with Cognito credentials
2. UI calls `GET /metrics` → shows available metrics in dropdown
3. User uploads test data → `POST /payloads` (gets presigned URL) → PUT to S3
4. User submits evaluation → `POST /evaluate` or `POST /invoke-evaluate`
5. Service starts Step Functions execution (parallel-by-dimension evaluation)
6. UI polls `GET /jobs/{jobId}` until status = COMPLETED (includes progress tracking)
7. UI displays results from the job response
8. User views history → `GET /experiments`

---

## Timeouts and Constraints

| Component | Timeout | Notes |
|-----------|---------|-------|
| Worker Lambda | 15 min | Single-dimension evaluation; bounded by Lambda max |
| Splitter Lambda | 30 sec | Lightweight: reads catalog.json, partitions metrics |
| Reducer Lambda | 5 min | Merges results, persists experiment |
| Step Functions state machine | 60 min | Allows multi-dimension evaluations to exceed 15-min Lambda limit |
| EKS container (HTTP) | No hard timeout | Bounded by K8s liveness/readiness probes |
| Liveness probe | every 30s (initial delay 15s) | Restarts pod if /health fails |
| Readiness probe | every 10s (initial delay 10s) | Removes pod from service if unhealthy |

---

## Multi-Architecture Builds

The `platforms` config controls which architectures are built:
- `"linux/amd64,linux/arm64"` — works on both Intel and Graviton EKS nodes (default)
- `"linux/amd64"` — Intel only (faster build)
- `"linux/arm64"` — Graviton only (cheapest runtime)

Docker buildx creates a manifest list in ECR so Kubernetes pulls the correct image for the node architecture automatically.

---

## Environment Variables (Container)

| Variable | Description |
|----------|-------------|
| `COGNITO_USER_POOL_ID` | Cognito User Pool ID for JWT validation |
| `COGNITO_APP_CLIENT_ID` | Cognito app client ID |
| `COGNITO_REGION` | AWS region of the Cognito pool |
| `COGNITO_DOMAIN` | Cognito domain URL |
| `AUTH_MODE` | `"existing"` or `"new"` |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | AWS region for SDK calls |
| `JOBS_TABLE_NAME` | DynamoDB table for job state |
| `PAYLOAD_BUCKET_NAME` | S3 bucket for request/result payloads |
| `UAEF_DYNAMODB_TABLE` | DynamoDB table for experiment persistence |
| `UAEF_S3_BUCKET` | S3 bucket for full results |
| `UAEF_EXPECTED_VERSION` | Expected uaef library version (mismatch guard) |
| `WORKER_FUNCTION_NAME` | Lambda function name for direct Worker invocations |
| `WORKER_MODE` | `"inline"` (in-process) or `"lambda"` (via real Lambda) — auto-set based on `EVAL_STATE_MACHINE_ARN` |
| `EVAL_STATE_MACHINE_ARN` | Step Functions ARN for evaluation orchestration (if set, WORKER_MODE becomes `"lambda"`) |
| `UAEF_ALLOWED_DATA_BUCKETS` | Comma-separated extra S3 bucket names `invoke-evaluate`'s `s3_data_path` may reference, beyond the payload bucket (default: none) |
| `UAEF_REDACT_PII_ON_PERSIST` | Set to `true` to redact EMAIL/PHONE/SSN/CREDIT_CARD spans in the persisted/UI-displayed copy of queries, agent responses, and judge reasoning (default: off — no redaction, matching prior behavior). Applied strictly after judge scoring; never changes evaluation results. NAME/ADDRESS are not covered (unimplemented ML-based detection). |
