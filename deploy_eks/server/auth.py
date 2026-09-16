# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cognito JWT validation middleware for the containerized UAEF Service.

Validates the Authorization Bearer token against the Cognito User Pool's
JWKS (JSON Web Key Set). Extracts the `sub` claim and injects it into the
request state for downstream route handlers.

This replaces the API Gateway Cognito authorizer in the Lambda setup.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import httpx
import jwt
from jwt import PyJWK, PyJWKClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Paths that skip JWT validation
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json"}


#: Set COGNITO_USER_POOL_ID unset AND this to "true" to explicitly run the
#: containerized service without JWT validation (e.g. local `docker run`
#: smoke-testing). Security review follow-up: previously, an unset
#: COGNITO_USER_POOL_ID silently fell back to treating every request as an
#: authenticated call from sub="anonymous" — not just unauthenticated, but a
#: single shared identity for every caller, which defeats the per-job
#: createdBy ownership check (the same check H-02 asks the Lambda path to add
#: to the experiment endpoints) for every route on this deployment path.
_ALLOW_NO_AUTH_ENV = "UAEF_EKS_ALLOW_NO_AUTH"


def _get_cognito_config() -> Dict[str, str]:
    """Read Cognito configuration from environment variables."""
    region = os.environ.get("COGNITO_REGION", "us-east-1")
    pool_id = os.environ.get("COGNITO_USER_POOL_ID", "")
    client_id = os.environ.get("COGNITO_APP_CLIENT_ID", "")
    return {
        "region": region,
        "pool_id": pool_id,
        "client_id": client_id,
        "issuer": f"https://cognito-idp.{region}.amazonaws.com/{pool_id}",
        "jwks_url": f"https://cognito-idp.{region}.amazonaws.com/{pool_id}/.well-known/jwks.json",
    }


def _allow_no_auth() -> bool:
    return os.environ.get(_ALLOW_NO_AUTH_ENV, "").strip().lower() in {"true", "1", "yes"}


class _JWKSCache:
    """Simple in-memory cache for the Cognito JWKS keys."""

    def __init__(self):
        self._keys: Optional[Dict[str, Any]] = None
        self._fetched_at: float = 0
        self._ttl: float = 3600  # refresh keys every hour

    def get_keys(self, jwks_url: str) -> Dict[str, Any]:
        now = time.time()
        if self._keys is None or (now - self._fetched_at) > self._ttl:
            self._refresh(jwks_url)
        return self._keys

    def _refresh(self, jwks_url: str) -> None:
        try:
            resp = httpx.get(jwks_url, timeout=10)
            resp.raise_for_status()
            self._keys = resp.json()
            self._fetched_at = time.time()
            logger.info("Refreshed JWKS keys from %s", jwks_url)
        except Exception:
            logger.exception("Failed to fetch JWKS from %s", jwks_url)
            if self._keys is None:
                self._keys = {"keys": []}


_jwks_cache = _JWKSCache()


def _verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify a Cognito JWT and return its claims, or None on failure.

    Fails closed: with COGNITO_USER_POOL_ID unset, JWT validation cannot be
    performed — this returns None (rejected) rather than fabricating a shared
    "anonymous" identity for every caller, unless UAEF_EKS_ALLOW_NO_AUTH is
    explicitly set (see CognitoJWTMiddleware.dispatch, which handles that
    opt-out path before this function is ever called).
    """
    config = _get_cognito_config()

    if not config["pool_id"]:
        logger.error(
            "COGNITO_USER_POOL_ID is not set — cannot validate JWTs. Rejecting."
        )
        return None

    jwks = _jwks_cache.get_keys(config["jwks_url"])

    # Decode the token header to find the key ID (kid)
    try:
        headers = jwt.get_unverified_header(token)
    except jwt.exceptions.DecodeError:
        return None

    kid = headers.get("kid")
    if not kid:
        return None

    # Find the matching key
    key_data = None
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            key_data = k
            break

    if key_data is None:
        logger.warning("No matching key found for kid=%s", kid)
        return None

    # Build the signing key from JWK data
    try:
        signing_key = PyJWK(key_data).key
    except Exception as exc:
        logger.warning("Failed to construct signing key: %s", exc)
        return None

    # Verify the token
    try:
        decode_options: Dict[str, Any] = {}
        decode_kwargs: Dict[str, Any] = {
            "algorithms": ["RS256"],
            "issuer": config["issuer"],
            "options": decode_options,
        }

        if config["client_id"]:
            decode_kwargs["audience"] = config["client_id"]
        else:
            decode_options["verify_aud"] = False

        claims = jwt.decode(token, signing_key, **decode_kwargs)
        return claims
    except jwt.exceptions.InvalidTokenError as exc:
        logger.warning("JWT validation failed: %s", exc)
        return None


class CognitoJWTMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that validates Cognito JWTs on protected routes.

    Fails closed: if COGNITO_USER_POOL_ID is unset, every request is rejected
    with 503 unless UAEF_EKS_ALLOW_NO_AUTH is explicitly set, making that a
    conscious opt-in (local `docker run` testing only) rather than a silent
    default that fabricates a single shared identity for every caller.
    """

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths and OPTIONS (CORS preflight)
        if request.url.path in _PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        if not _get_cognito_config()["pool_id"]:
            if not _allow_no_auth():
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": (
                            "This server is not configured for authentication. "
                            "Set COGNITO_USER_POOL_ID, or set "
                            f"{_ALLOW_NO_AUTH_ENV}=true to explicitly run "
                            "without auth (local testing only)."
                        )
                    },
                )
            # Explicit opt-out: every caller shares one identity. Fine for a
            # single developer's local smoke test; never set this in a shared
            # deployment — every job/experiment becomes visible to anyone
            # hitting this instance, same as the unconfigured default used to
            # be, just now an explicit choice instead of a silent one.
            request.state.caller_sub = "local-dev-no-auth"
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Missing or malformed Authorization header"},
            )

        token = auth_header[7:]  # Strip "Bearer "
        claims = _verify_token(token)
        if claims is None:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or expired token"},
            )

        # Inject the caller's sub into request state
        request.state.caller_sub = claims.get("sub", "unknown")
        return await call_next(request)
