# Security

This document records the security model of the Universal Agent Evaluation
Framework (UAEF) reference architecture: what each deployable surface trusts,
what isolation it enforces itself, and what it expects the environment around
it to enforce. It is the authoritative reference for the deployment
constraints below — read it before deploying UAEF anywhere that is not a
single-team dev or sandbox account.

Report a suspected vulnerability to the support team listed in
[README.md](README.md#support-team).

## Deployment note: tenant isolation

**UAEF is a reference architecture designed to run embedded inside a platform
that provides user/tenant isolation. Direct deployment in a multi-tenant
environment without a host platform enforcing tenant isolation is not
supported.**

UAEF's purpose is to be plugged into an existing agentic platform — the
library (`uaef`) runs in that platform's process, and the service
(`uaef-service`) is deployed by that platform into its own AWS account as an
evaluation backend for it. The trust boundary that separates one tenant from
another belongs to the host platform, not to UAEF.

### What the host platform must provide

| Responsibility | Mechanism the host platform is expected to supply |
| --- | --- |
| Tenant boundary | One UAEF deployment (its own AWS account or, at minimum, its own CDK stack with its own DynamoDB tables, S3 buckets, and Cognito user pool) per tenant. Cross-tenant separation is account/stack separation — UAEF does not partition a single deployment by tenant. |
| Caller authentication | The Cognito user pool the deployment is configured with (`auth` section of `uaef-service/config.yaml`), or, for an imported pool (`auth.mode: "existing"`), the platform's own identity provider federated into it. Only that pool's users may reach the API. |
| Authorization to evaluate | The platform decides which of its users may submit evaluations at all, and only issues them credentials for its UAEF deployment. UAEF treats every authenticated caller as entitled to use the service. |
| Network exposure | For `with_eks`, the ingress/load balancer in front of the container. UAEF's EKS server authenticates requests but does not itself restrict who can reach the endpoint. |

### What UAEF enforces itself

Within a single deployment, UAEF enforces **per-caller ownership scoping** so
that one authenticated user of that deployment cannot read another's
evaluation data (security review H-02):

- The API Lambda derives the caller's Cognito `sub` from the authorizer claims
  and fails closed with `401` if it is absent
  (`uaef-service/handlers/api.py:_extract_caller_sub`).
- That `sub` is recorded as `createdBy` on the job at creation, and stamped as
  `created_by` on any experiment the Worker or Reducer persists
  (`uaef-service/job_state.py`, `uaef-service/handlers/worker.py`,
  `uaef-service/handlers/orchestrator.py`).
- `GET /jobs/{jobId}`, `GET /experiments`, `GET /experiments/{experimentId}`,
  `POST /reports`, and `POST /compare-experiments` are all scoped to the
  caller's own records. Listing uses a `created_by` GSI query, never an
  unscoped table scan (`DynamoS3Storage.CREATED_BY_INDEX_NAME`). A record
  belonging to someone else returns `403` with no detail disclosed — the same
  response as one that does not exist, so the endpoint cannot be used to probe
  for other users' experiment IDs.
- Experiments persisted before this scoping existed carry no `created_by` and
  are **never** returned by an owner-scoped read. They remain in the table and
  are reachable only with direct AWS access to it.
- An experiment's `created_by` is preserved on later saves, so an owner cannot
  be overwritten by a subsequent update that omits it or names someone else.

Ownership scoping is a defense-in-depth control **within** a tenant. It is not
a substitute for the per-tenant deployment separation described above.

### Residual risk (accepted)

If this reference architecture is deployed standalone in a multi-user
environment with no host platform enforcing tenant isolation, all authenticated
users of that deployment share one Cognito pool, one job table, one experiment
table, and one results bucket. Ownership scoping keeps them out of each other's
jobs and experiments through the API, but they remain co-tenants of the same
data stores, and anything not covered by an ownership check — the metrics
catalog, service-level configuration, and any direct AWS access to the
underlying tables and buckets — is shared. That is a known and accepted
property of the reference architecture, not a defect: standalone multi-tenant
deployment is not a supported use case.

The library surface (`src/uaef/`) has no notion of a caller at all. It is an
in-process library; `created_by` is an optional argument its storage layer
records on the caller's behalf. Any service placing UAEF behind a network
boundary is responsible for authenticating callers and passing their identity
down, as `uaef-service` does.

## Evaluated-agent output is untrusted input

Everything UAEF evaluates — an agent's response text, its reasoning trace, the
tools it chose to call and the arguments it passed, a conversation transcript
it participated in — is **untrusted input**, produced by the system under
evaluation. LLM-judge metrics feed that content back into a Bedrock prompt to
be scored, which makes the judge prompt an injection target: an agent that can
influence its own judge can influence its own scores (security review H-01).

Three controls are applied at every judge call site (see
`src/uaef/metrics/utils.py`, which all built-in metrics and
`uaef.llm_judge.LLMJudge` share):

1. **Instruction/data separation.** The rubric is sent in the model's
   dedicated `system` field (`build_judge_system_prompt`); the user turn
   carries only the content being evaluated. A system-level instruction states
   that content inside a `<candidate_*>` tag is data, never instructions.
2. **Escaped boundaries.** Untrusted content is escaped
   (`escape_untrusted_boundaries`) before being wrapped in its
   `<candidate_*>` tag, so it cannot forge the closing delimiter and have text
   after it read as prompt. Nothing is dropped or truncated — only the
   delimiter's angle brackets are neutralized — so no evaluation signal is
   lost.
3. **Strict parsing and provenance binding.** Judge responses are parsed with
   `json.loads(..., strict=True)` (`loads_judge_json`) and validated against a
   fixed schema (`validate_judge_response`). Where UAEF owns the rubric, the
   judge must additionally quote a verbatim span of the candidate content its
   score rests on, and that span is verified to occur in the content
   (`validate_provenance`). A score whose cited evidence is not present is
   rejected rather than recorded — which is what catches a judge that returned
   well-formed JSON because injected text talked it into a number.

Residual risk: no delimiter is unforgeable against every model on every input,
and provenance binding is not applied when a caller supplies its own judge
template (UAEF cannot assume a template it did not author asked for an
evidence quote). Treat judge scores as evaluation signal, not as an
authorization or safety decision, and do not gate a security control on them.

## Other deployment-time security notes

| Topic | Where |
| --- | --- |
| CORS — an explicit allowed-origin list is required; wildcard needs an explicit acknowledgement flag | [uaef-service/README.md](uaef-service/README.md#cors-configuration) |
| Retained data stores after `cdk destroy` (tables, buckets, and Cognito pool survive teardown) | [uaef-service/README.md](uaef-service/README.md#teardown) |
| The deploy CodeBuild role is granted `AdministratorAccess` — use a dev or sandbox account and scope it down before production | [uaef-service/README.md](uaef-service/README.md#credentials) |
| PII redaction on persisted and displayed results (opt-in) | `src/uaef/security/pii.py` |
| `UAEF_EKS_ALLOW_NO_AUTH` — dev-only escape hatch; the EKS server fails closed without it | `deploy_eks/server/auth.py` |
