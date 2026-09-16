# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""UiStack — minimal deployed UI for the UAEF Service.

Hosts a tiny static single-page app on **S3 + CloudFront** and wires it to the
service's API Gateway with **Cognito Hosted-UI** login (OAuth authorization-code
+ PKCE). This mirrors the reference app's Amplify + Hosted-UI pattern, scaled
down to the essentials: sign in, submit a trace, poll the async job, show the
result.

When ``auth_mode=existing``, the Hosted-UI domain is expected to be managed
externally and must be passed via ``-c existing_cognito_domain=<base-url>``.
A new UI client is still created inside the imported pool unless
``existing_app_client_id`` is provided in the auth context.

Provisioned only when ``-c deploy_ui=true`` is passed (the parent gates it).
Part of the standalone ``uaef-service`` app; does NOT import ``uaef``.

Requirements: 12.1
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    NestedStack,
    RemovalPolicy,
    CfnOutput,
    Stack,
    aws_cognito as cognito,
    aws_s3 as s3,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_s3_deployment as s3deploy,
)
from constructs import Construct

from .constants import UI_DOMAIN_PREFIX

# The deployed UI is the built React app (Vite output). CodeBuild runs
# `npm ci && npm run build` in ui/app/ before `cdk deploy` (see buildspec.yml),
# producing ui/app/dist which CloudFront serves. config.json is injected at
# deploy time by the BucketDeployment below.
_WEB_DIR = str(Path(__file__).resolve().parent.parent / "ui" / "app" / "dist")


class UiStack(NestedStack):
    """Static SPA on S3+CloudFront with Cognito Hosted-UI login."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        user_pool: cognito.IUserPool,
        auth_mode: str = "new",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # This stack intentionally does NOT take the ApiStack. It used to bake
        # the API's URL into config.json here, which made UiStack depend on
        # ApiStack — while ApiStack (and StorageStack) needed *this* stack's
        # CloudFront origin for their CORS rules. That mutual dependency is a
        # cycle CloudFormation cannot deploy, so the old flow broke it outside
        # CloudFormation with a two-pass deploy (deploy once with a wildcard
        # origin, read the CloudFront domain that got created, redeploy with the
        # real origin). By dropping the API reference here, this stack depends
        # only on the User Pool, so its CloudFront origin can be handed to the
        # CORS rules as an ordinary CloudFormation reference and the whole thing
        # deploys in one pass with no wildcard. The API URL is written into
        # config.json afterwards by ConfigStack (see app.py), which is created
        # after both this stack and ApiStack and so can reference both without a
        # cycle.

        region = Stack.of(self).region

        # --- Static site bucket (private; served via CloudFront) ----------
        site_bucket = s3.Bucket(
            self,
            "SiteBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # --- CloudFront distribution (SPA: 403/404 -> index.html) ---------
        distribution = cloudfront.Distribution(
            self,
            "Distribution",
            default_root_object="index.html",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(site_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
            ),
            error_responses=[
                cloudfront.ErrorResponse(
                    http_status=403, response_http_status=200,
                    response_page_path="/index.html",
                ),
                cloudfront.ErrorResponse(
                    http_status=404, response_http_status=200,
                    response_page_path="/index.html",
                ),
            ],
        )
        ui_url = f"https://{distribution.distribution_domain_name}"

        # --- Cognito Hosted-UI domain + UI app client (OAuth code/PKCE) ---
        #
        # auth_mode=existing: The domain already exists externally. We just use
        #   the URL string passed via -c existing_cognito_domain. No CFN resource.
        # auth_mode=new: This stack creates and owns the domain resource.

        if auth_mode == "existing":
            existing_domain = self.node.try_get_context("existing_cognito_domain")
            if existing_domain:
                cognito_domain_base_url = existing_domain
            else:
                domain_prefix = self.node.try_get_context("ui_domain_prefix") or UI_DOMAIN_PREFIX
                cognito_domain_base_url = f"https://{domain_prefix}.auth.{region}.amazoncognito.com"
        else:
            domain_prefix = self.node.try_get_context("ui_domain_prefix") or UI_DOMAIN_PREFIX
            ui_domain = cognito.UserPoolDomain(
                self,
                "UiHostedDomain",
                user_pool=user_pool,
                cognito_domain=cognito.CognitoDomainOptions(domain_prefix=domain_prefix),
            )
            cognito_domain_base_url = ui_domain.base_url()

        ui_client = cognito.UserPoolClient(
            self,
            "UiClient",
            user_pool=user_pool,
            user_pool_client_name="uaef-service-ui-client",
            generate_secret=False,
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[
                    cognito.OAuthScope.OPENID,
                    cognito.OAuthScope.EMAIL,
                    cognito.OAuthScope.PROFILE,
                ],
                callback_urls=[ui_url],
                logout_urls=[ui_url],
            ),
        )

        # --- Deploy the static site ---------------------------------------
        # Only the built SPA assets here — NOT config.json. config.json carries
        # the API base URL, which lives in ApiStack; writing it here would
        # recreate the UiStack -> ApiStack dependency this stack was refactored
        # to avoid. ConfigStack writes config.json into this same bucket after
        # ApiStack exists, and (running last) owns the CloudFront invalidation.
        s3deploy.BucketDeployment(
            self,
            "DeploySite",
            sources=[s3deploy.Source.asset(_WEB_DIR)],
            destination_bucket=site_bucket,
        )

        # Expose the handles ConfigStack needs to write config.json and the
        # CORS origin StorageStack/ApiStack lock their rules to.
        self.site_bucket = site_bucket
        self.distribution = distribution
        self.ui_client = ui_client
        self.cognito_domain_base_url = cognito_domain_base_url
        self.ui_url = ui_url
        #: The browser Origin of the deployed UI (scheme + host, no trailing
        #: slash) — exactly what a browser sends in the ``Origin`` header and
        #: what the API/payload-bucket CORS rules allow. A CloudFormation token
        #: resolved at deploy time, so consumers get the real value in one pass.
        self.ui_origin = ui_url

        CfnOutput(self, "UiUrl", value=ui_url, description="Deployed UI URL")
        CfnOutput(
            self, "UiLoginDomain", value=cognito_domain_base_url,
            description="Cognito Hosted-UI base URL",
        )
