# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CDK infrastructure package for the standalone ``uaef-service`` app.

This package contains the AWS CDK nested stacks that make up the UAEF Service:

* :mod:`infra.auth_stack` — Cognito User Pool + app client.
* :mod:`infra.storage_stack` — Jobs DynamoDB table + payload bucket.
* :mod:`infra.api_stack` — API Gateway REST + Cognito authorizer.
* :mod:`infra.worker_stack` — Worker Lambda (``uaef[server]``).
* :mod:`infra.ui_stack` — optional Amplify UI hosting.

The parent ``UaefServiceStack`` (see ``uaef-service/app.py``) orchestrates these
nested stacks and wires shared configuration between them.

Per the design's "Single Source of Truth" requirement, this app NEVER vendors,
copies, or re-implements UAEF library source. The Worker Lambda installs the
published ``uaef`` package pinned to the exact version recorded in
:mod:`infra.constants`.
"""
