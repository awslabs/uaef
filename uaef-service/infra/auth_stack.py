# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuthStack — Cognito User Pool + app client for the UAEF Service.

This nested stack provisions (or imports) the identity layer that fronts the
service's protected API routes. API Gateway uses this User Pool as a Cognito
authorizer: every protected route requires a Cognito-issued JWT with a valid
signature, an unexpired ``exp`` claim, and an issuer matching this pool
(Requirement 9.1).

Two modes are supported via CDK context:

    1. **Create new** (default) — a fresh Cognito User Pool and app client are
       provisioned by this stack (original behavior).
    2. **Use existing** — an external Cognito User Pool (e.g. one owned by the
       platform team) is imported by ARN/ID. No new pool is created. An
       existing app client ID can optionally be provided; if omitted, a new
       client is created inside the imported pool.

Context keys (``-c key=value`` on ``cdk deploy``):

    auth_mode             "new" (default) | "existing"
    existing_user_pool_id Required when auth_mode=existing. The pool ID.
    existing_user_pool_arn
                          Optional (but recommended) when using an existing
                          pool. Enables proper ARN-based references.
    existing_app_client_id
                          Optional. If provided, no new client is created.

The ``sub`` claim of the authenticated caller is later recorded as a job's
``createdBy`` value so callers can only read jobs they created.

This module is part of the standalone ``uaef-service`` deployable app and does
NOT import the ``uaef`` package.

Requirements: 9.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aws_cdk import (
    NestedStack,
    RemovalPolicy,
    Duration,
    aws_cognito as cognito,
)
from constructs import Construct

from .constants import USER_POOL_NAME, USER_POOL_CLIENT_NAME


@dataclass
class ExistingAuthConfig:
    """Configuration for importing an existing Cognito User Pool."""

    user_pool_id: str
    user_pool_arn: Optional[str] = None
    app_client_id: Optional[str] = None


class AuthStack(NestedStack):
    """Cognito User Pool and app client used by the API Gateway authorizer.

    Supports two modes:
        - **new** (default): Creates a new User Pool and app client.
        - **existing**: Imports an externally-managed User Pool by ID/ARN.
          Optionally imports an existing app client or creates a new one in
          the imported pool.

    Attributes:
        user_pool: The Cognito User Pool issuing JWTs for the service. Its
            issuer URL and JWKS back the API Gateway Cognito authorizer.
        user_pool_client: The app client used by UIs/SDKs to obtain tokens.
            May be None if an existing client ID was provided (the IUserPoolClient
            import does not expose all attributes).
        auth_mode: "new" or "existing" — indicates which mode was used.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        existing_auth: Optional[ExistingAuthConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Resolve auth mode from explicit param or CDK context.
        if existing_auth is not None:
            self._auth_mode = "existing"
            self._existing = existing_auth
        else:
            ctx_mode = self.node.try_get_context("auth_mode") or "new"
            if ctx_mode == "existing":
                pool_id = self.node.try_get_context("existing_user_pool_id")
                if not pool_id:
                    raise ValueError(
                        "auth_mode=existing requires 'existing_user_pool_id' context value. "
                        "Pass it via: cdk deploy -c auth_mode=existing "
                        "-c existing_user_pool_id=<pool-id>"
                    )
                self._auth_mode = "existing"
                self._existing = ExistingAuthConfig(
                    user_pool_id=pool_id,
                    user_pool_arn=self.node.try_get_context("existing_user_pool_arn"),
                    app_client_id=self.node.try_get_context("existing_app_client_id"),
                )
            else:
                self._auth_mode = "new"
                self._existing = None

        if self._auth_mode == "existing":
            self._import_existing_pool()
        else:
            self._create_new_pool()

    # ------------------------------------------------------------------ #
    # Mode: create new pool (original behavior)
    # ------------------------------------------------------------------ #

    def _create_new_pool(self) -> None:
        """Provision a brand-new Cognito User Pool and app client."""
        # Email is the sign-in identifier; self sign-up is disabled so operators
        # control membership. Strong password policy and account recovery via
        # email. The pool's issuer + JWKS are what the API authorizer validates.
        self.user_pool = cognito.UserPool(
            self,
            "UserPool",
            user_pool_name=USER_POOL_NAME,
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True, username=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=True),
            ),
            password_policy=cognito.PasswordPolicy(
                min_length=12,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=True,
            ),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # No generated client secret (suitable for public UI / SDK clients that
        # cannot keep a secret). SRP + admin user/password auth flows are
        # enabled. Token validity is bounded; refresh tokens are longer-lived.
        self.user_pool_client = self.user_pool.add_client(
            "AppClient",
            user_pool_client_name=USER_POOL_CLIENT_NAME,
            generate_secret=False,
            auth_flows=cognito.AuthFlow(
                user_srp=True,
                user_password=True,
                admin_user_password=True,
            ),
            access_token_validity=Duration.hours(1),
            id_token_validity=Duration.hours(1),
            refresh_token_validity=Duration.days(30),
            prevent_user_existence_errors=True,
        )

        self._user_pool_client_id = self.user_pool_client.user_pool_client_id

    # ------------------------------------------------------------------ #
    # Mode: import existing pool
    # ------------------------------------------------------------------ #

    def _import_existing_pool(self) -> None:
        """Import an externally-managed Cognito User Pool (and optionally client)."""
        cfg = self._existing
        if cfg is None:
            raise ValueError(
                "_import_existing_pool() requires existing auth config, but "
                "self._existing is None (auth mode is not 'existing')"
            )

        # Import the existing User Pool. If ARN is provided, use the full
        # from_user_pool_arn; otherwise fall back to from_user_pool_id.
        if cfg.user_pool_arn:
            self.user_pool = cognito.UserPool.from_user_pool_arn(
                self, "ImportedUserPool", user_pool_arn=cfg.user_pool_arn
            )
        else:
            self.user_pool = cognito.UserPool.from_user_pool_id(
                self, "ImportedUserPool", user_pool_id=cfg.user_pool_id
            )

        # If an existing app client ID is provided, import it; otherwise create
        # a new client in the imported pool so the UAEF UI has its own client.
        if cfg.app_client_id:
            self.user_pool_client = cognito.UserPoolClient.from_user_pool_client_id(
                self, "ImportedAppClient", user_pool_client_id=cfg.app_client_id
            )
            self._user_pool_client_id = cfg.app_client_id
        else:
            # Create a new app client inside the existing pool for UAEF's use.
            self.user_pool_client = cognito.UserPoolClient(
                self,
                "AppClient",
                user_pool=self.user_pool,
                user_pool_client_name=USER_POOL_CLIENT_NAME,
                generate_secret=False,
                auth_flows=cognito.AuthFlow(
                    user_srp=True,
                    user_password=True,
                    admin_user_password=True,
                ),
                access_token_validity=Duration.hours(1),
                id_token_validity=Duration.hours(1),
                refresh_token_validity=Duration.days(30),
                prevent_user_existence_errors=True,
            )
            self._user_pool_client_id = self.user_pool_client.user_pool_client_id

    # ------------------------------------------------------------------ #
    # Public properties
    # ------------------------------------------------------------------ #

    @property
    def auth_mode(self) -> str:
        """Returns 'new' or 'existing' indicating which mode is active."""
        return self._auth_mode

    @property
    def user_pool_id(self) -> str:
        """The User Pool id (consumed by the API stack's authorizer)."""
        return self.user_pool.user_pool_id

    @property
    def user_pool_client_id(self) -> str:
        """The app client id (consumed by clients and the API stack)."""
        return self._user_pool_client_id
