# UAEF Demo UI

React + FastAPI demo for running evaluations against live agents.

## Quick Start

### Backend

```bash
cd demo/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ../../
# The backend uses your AMBIENT AWS credentials (boto3 default provider chain):
# environment variables, `aws configure` / a named profile, SSO, or an IAM role.
# Set them in this shell BEFORE starting uvicorn — the demo does not accept
# credentials over the API. GET /api/aws-identity shows the resolved identity.
#
# The server fails closed on auth: it refuses every request with a 503 until
# you either set UAEF_DEMO_API_KEY or explicitly opt out of auth. For local
# development, opt out explicitly:
export UAEF_DEMO_ALLOW_NO_AUTH=true
python -m uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd demo/frontend
npm install
npm run dev
```


Open http://localhost:3000

## Security notes (read before exposing beyond localhost)

This demo is intended to run **locally**. The backend fails closed by
default: it rejects every request with a 503 unless you configure auth (or
explicitly opt out for local development). If you must serve it beyond your
machine:

- **API authentication** — set `UAEF_DEMO_API_KEY` on the backend to require an
  `X-API-Key` header on every request. Build the frontend with a matching
  `VITE_DEMO_API_KEY` so it sends the header. For local development only, you
  can instead set `UAEF_DEMO_ALLOW_NO_AUTH=true` to explicitly run without
  auth — never set this on a shared or exposed host.
- **CORS** — set `CORS_ALLOWED_ORIGINS` (comma-separated) to your UI origin(s).
  It defaults to `*` for local dev only; lock it down for any deployment.
- **AWS credentials** — provided via the backend's ambient credential chain
  (env/profile/SSO/role); they are never accepted over the API.
- **Transport** — terminate TLS in front of the backend (reverse proxy) so
  the API key and traffic are never sent in plaintext.

## How It Works

1. Upload a ground truth file (xlsx/csv) with `Question` and `Answer` columns
2. Select your agent framework (Bedrock, LangGraph, LangChain, etc.) and provide connection details
3. Choose which metrics to evaluate (28 available across 7 dimensions)
4. Optionally enable persistence to DynamoDB + S3
5. Click "Run Evaluation" — UAEF sends each question to your agent, collects the trace, adapts it, and scores it
6. View results: overall score, per-metric averages, dimension radar, and per-test-case detail with agent responses
