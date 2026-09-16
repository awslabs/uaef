# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""UAEF Service — FastAPI HTTP server wrapping Lambda handler logic.

This is the containerized equivalent of the API Gateway + Lambda setup.
It exposes the same REST endpoints, validates Cognito JWTs via middleware,
and invokes the worker logic in-process (instead of async Lambda invoke).

Environment variables:
    COGNITO_USER_POOL_ID    - Cognito User Pool ID for JWT validation. Required
                              unless UAEF_EKS_ALLOW_NO_AUTH=true (local testing
                              only) — the server fails closed with 503 on every
                              request when neither is set.
    UAEF_EKS_ALLOW_NO_AUTH  - Set to "true" to explicitly run without JWT
                              validation (every caller shares one identity;
                              local `docker run` smoke-testing only, never a
                              shared deployment).
    COGNITO_APP_CLIENT_ID   - (optional) restrict tokens to this client
    COGNITO_REGION          - AWS region of the Cognito pool (default: us-east-1)
    AUTH_MODE               - "existing" or "new"
    JOBS_TABLE_NAME         - DynamoDB table for job state
    PAYLOAD_BUCKET_NAME     - S3 bucket for request/result payloads
    UAEF_DYNAMODB_TABLE     - DynamoDB table for UAEF experiment persistence
    UAEF_S3_BUCKET          - S3 bucket for UAEF full results
    UAEF_EXPECTED_VERSION   - Expected uaef library version
    WORKER_MODE             - "inline" (default) runs worker in-process
    CORS_ALLOWED_ORIGINS    - Comma-separated allowed CORS origins (default: "*")
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .auth import CognitoJWTMiddleware
from .routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="UAEF Service",
    description="Universal Agent Evaluation Framework — containerized backend",
    version="0.1.0",
)

# CORS — origins are configurable via the CORS_ALLOWED_ORIGINS env var
# (comma-separated). Defaults to "*" so behavior is unchanged unless an operator
# sets it. The EKS "bring your own UI" mode does not know the UI origin at build
# time, so operators lock down by setting e.g.
# CORS_ALLOWED_ORIGINS=https://my-ui.example.com in the deployment.
_cors_allowed_origins = [
    o.strip()
    for o in os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allowed_origins,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
)

# Cognito JWT validation middleware
app.add_middleware(CognitoJWTMiddleware)

# Mount all routes
app.include_router(router)


@app.get("/health")
async def health():
    """Health check endpoint for EKS liveness/readiness probes."""
    return {"status": "healthy"}
