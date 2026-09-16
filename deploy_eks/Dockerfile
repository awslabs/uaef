# UAEF Service — Containerized HTTP server for EKS deployment.
#
# This container wraps the Lambda handler logic as a persistent FastAPI server.
# It includes both the API routing layer AND the worker (evaluation) logic,
# running in-process instead of via async Lambda invocation.
#
# Build:
#   docker build --platform linux/amd64 -t uaef-service-eks .
#
# Run locally:
#   docker run -p 8080:8080 \
#     -e COGNITO_USER_POOL_ID=us-east-1_xxxxx \
#     -e COGNITO_APP_CLIENT_ID=xxxxx \
#     -e COGNITO_REGION=us-east-1 \
#     -e AUTH_MODE=existing \
#     -e JOBS_TABLE_NAME=uaef-service-jobs \
#     -e PAYLOAD_BUCKET_NAME=uaef-service-payloads \
#     -e UAEF_DYNAMODB_TABLE=uaef-experiments \
#     -e UAEF_S3_BUCKET=uaef-results \
#     uaef-service-eks

# Pinned by digest (security review M-07): even a patch-pinned tag like
# `3.12.8-slim` can be re-pushed to point at a different image; the digest
# cannot. This is the manifest-list digest for `python:3.12.8-slim` as of
# this change — a multi-arch manifest list, so `docker buildx build
# --platform linux/amd64,linux/arm64` (see deploy.sh) still resolves the
# correct per-architecture image from it. Re-pin deliberately when bumping
# the base image (`docker manifest inspect python:3.12.8-slim -v` or the
# registry API) rather than reverting to a floating/mutable tag.
FROM python:3.12.8-slim@sha256:2199a62885a12290dc9c5be3ca0681d367576ab7bf037da120e564723292a2f0 AS builder

WORKDIR /app

# System deps for native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in layers for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install heavy ML/eval dependencies (layered for Docker cache efficiency)
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.14.1
RUN pip install --no-cache-dir pandas==2.2.3 pyarrow==18.1.0
RUN pip install --no-cache-dir scikit-learn==1.6.0
RUN pip install --no-cache-dir langchain==0.3.13 langchain-core==0.3.28 langchain-community==0.3.13 langchain-aws==0.2.10 langchain-openai==0.2.14
RUN pip install --no-cache-dir langgraph==0.2.60 langgraph-checkpoint==2.0.10 langgraph-prebuilt==0.1.2 langgraph-sdk==0.1.51
RUN pip install --no-cache-dir datasets==3.2.0 huggingface-hub==0.27.0
RUN pip install --no-cache-dir deepeval==2.1.4 openai==1.58.1
RUN pip install --no-cache-dir ragas==0.2.8

# Install UAEF library (same pinned version as Lambda)
COPY wheels/ ./wheels/
RUN pip install --no-cache-dir ./wheels/uaef-*.whl

# Force FastAPI/Starlette compatibility AFTER all deps are installed.
# uaef[server] can pull in a newer starlette that breaks FastAPI 0.115.x
# (Starlette 0.46+ removed on_startup from Router.__init__).
RUN pip install --no-cache-dir --force-reinstall fastapi==0.115.6 starlette==0.41.3

# --- Runtime stage ---
# Pinned by digest — same rationale and digest as the builder stage above.
FROM python:3.12.8-slim@sha256:2199a62885a12290dc9c5be3ca0681d367576ab7bf037da120e564723292a2f0

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy server code
COPY server/ ./server/

# Copy handler code (synced from uaef-service)
COPY uaef-service/ ./uaef-service/

# Ensure handler imports work
ENV PYTHONPATH="/app/uaef-service:/app"

# Create non-root user and switch to it
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Container health probe against the app's /health endpoint. Kubernetes uses its
# own liveness/readiness probes; this HEALTHCHECK also covers plain `docker run`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=3)" || exit 1

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
