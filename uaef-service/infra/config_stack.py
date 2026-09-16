# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConfigStack — writes the deployed UI's runtime ``config.json``.

The SPA fetches ``/config.json`` at load time (see ``ui/app/src/config.js``) to
learn its backend API URL and Cognito settings. That file needs values from two
different stacks:

* the **API base URL** from :class:`ApiStack`, and
* the **CloudFront/S3 site bucket**, **Cognito app client**, and **Hosted-UI
  domain** from :class:`UiStack`.

Writing it from UiStack would make UiStack depend on ApiStack, while ApiStack
(and StorageStack) already depend on UiStack's CloudFront origin for their CORS
rules — a dependency cycle CloudFormation cannot deploy, and the reason the old
flow needed a two-pass "deploy with a wildcard origin, then redeploy with the
real one" dance.

This stack breaks that: it is created **after** both UiStack and ApiStack (see
``app.py``), so it can reference both. Nothing references it back, so the graph
stays acyclic and the whole service — including the real, non-wildcard CORS
origin — deploys in a single pass.
"""

from __future__ import annotations

from aws_cdk import (
    NestedStack,
    Stack,
    aws_cognito as cognito,
    aws_s3 as s3,
    aws_cloudfront as cloudfront,
    aws_s3_deployment as s3deploy,
)
from constructs import Construct


class ConfigStack(NestedStack):
    """Deploys the SPA's runtime ``config.json`` into the UI site bucket."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        site_bucket: s3.IBucket,
        distribution: cloudfront.IDistribution,
        user_pool: cognito.IUserPool,
        ui_client: cognito.IUserPoolClient,
        cognito_domain_base_url: str,
        ui_url: str,
        api_base_url: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        region = Stack.of(self).region

        # config.json only — the built SPA assets are deployed by UiStack. Two
        # BucketDeployments target the same bucket, so this one MUST NOT prune
        # (pruning would delete the site UiStack uploaded). This stack depends
        # on UiStack (via ``site_bucket``) and so runs after it; it owns the
        # CloudFront invalidation because it is the last writer to the bucket.
        s3deploy.BucketDeployment(
            self,
            "DeployConfig",
            sources=[
                s3deploy.Source.json_data(
                    "config.json",
                    {
                        "region": region,
                        "userPoolId": user_pool.user_pool_id,
                        "clientId": ui_client.user_pool_client_id,
                        "cognitoDomain": cognito_domain_base_url,
                        "apiBaseUrl": api_base_url,
                        "redirectUri": ui_url,
                    },
                ),
            ],
            destination_bucket=site_bucket,
            prune=False,
            distribution=distribution,
            distribution_paths=["/*"],
        )
