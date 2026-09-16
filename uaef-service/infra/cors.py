# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CORS origin resolution (security review M-01).

Both :class:`ApiStack` (API Gateway preflight) and :class:`StorageStack` (the
payload S3 bucket) must allow exactly the same browser origins, or the SPA hits
CORS failures on one surface but not the other. This helper is the single place
that computes that allow-list so the two can never drift.

There are two ways an origin is supplied:

* **Derived (with_ui).** The parent passes ``cors_origin`` — the deployed UI's
  own CloudFront origin, as a CloudFormation token resolved at deploy time. The
  real origin is therefore known within a single deploy, so no wildcard is ever
  needed. Additional origins from the ``api_cors_allowed_origins`` context (for
  example a ``http://localhost:5173`` dev server) are merged in; a stray ``*``
  there is dropped, because a concrete UI origin is authoritative.

* **Explicit (with_eks / UI-less API-only).** No CloudFront origin exists to
  derive, so the origin(s) MUST be supplied via the ``api_cors_allowed_origins``
  context. A ``*`` wildcard is rejected unless the deployer explicitly opts in
  with ``i_acknowledge_insecure_cors=true`` — an escape hatch that with_ui no
  longer needs.
"""

from __future__ import annotations

from typing import List, Optional

from constructs import Construct

_CONTEXT_KEY = "api_cors_allowed_origins"
_ACK_KEY = "i_acknowledge_insecure_cors"


def resolve_cors_origins(scope: Construct, cors_origin: Optional[str]) -> List[str]:
    """Return the CORS allow-list for ``scope``'s stack.

    Args:
        scope: The stack reading CDK context.
        cors_origin: The derived UI origin (a CloudFormation token) when the UI
            is deployed in the same run, else ``None``.

    Returns:
        The list of allowed origins. Contains ``"*"`` only on the explicit path
        when the deployer has acknowledged an insecure wildcard.

    Raises:
        ValueError: On the explicit path when no origin is supplied, or a
            wildcard is supplied without acknowledgement.
    """
    ctx = scope.node.try_get_context(_CONTEXT_KEY)
    extras = [o.strip() for o in str(ctx).split(",") if o.strip()] if ctx else []

    if cors_origin is not None:
        # Derived path: the real UI origin is authoritative. Merge any extra
        # origins from context, dropping a wildcard (meaningless here).
        return [cors_origin] + [o for o in extras if o != "*"]

    # Explicit path: an origin must be provided.
    if not extras:
        raise ValueError(
            f"{_CONTEXT_KEY} CDK context is required (security review M-01) when "
            "the UI is not deployed in this run — pass the calling client's "
            "origin(s), e.g.\n"
            f"  cdk deploy -c {_CONTEXT_KEY}=https://my-ui.example.com\n"
            "With the built-in UI (deploy_ui=true) this is derived automatically "
            "and need not be set."
        )
    if "*" in extras:
        ack = str(scope.node.try_get_context(_ACK_KEY) or "").strip().lower() in (
            "true",
            "1",
            "yes",
        )
        if not ack:
            raise ValueError(
                f"{_CONTEXT_KEY}='*' allows any web origin to make credentialed "
                "calls against this Cognito-protected API (security review M-01) "
                f"and is rejected by default. Pass -c {_ACK_KEY}=true to opt in "
                "explicitly, or supply the real origin(s) instead."
            )
    return extras
