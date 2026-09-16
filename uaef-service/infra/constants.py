# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for the UAEF Service CDK app — the single source of truth.

Resource naming and deploy-time configuration are loaded from
``uaef-service/config.yaml`` so operators can change deployment identity without
touching Python.  Version-pinning logic and the derived pip spec remain here
because they participate in runtime guards (worker version-mismatch check).

Single Source of Truth (Requirements 2.1, 2.2):
    The UAEF Service declares ``uaef`` as a dependency pinned to a single exact
    version using an equality specifier (``==``). The service tree contains NO
    copy, vendored snapshot, or re-implementation of UAEF source — the Worker
    Lambda ``pip install``s this exact spec from the registry and ``import uaef``.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# --------------------------------------------------------------------------- #
# Load config.yaml (lives one level above this module: uaef-service/config.yaml)
# --------------------------------------------------------------------------- #

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with open(_CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

_DEPLOY_SUFFIX: str = _cfg["deploy_suffix"]

# --------------------------------------------------------------------------- #
# Single source of truth: the pinned UAEF library dependency.
# --------------------------------------------------------------------------- #

#: Exact UAEF library version the service is built and validated against.
UAEF_VERSION: str = _cfg["uaef"]["version"]

#: The extra the Worker Lambda installs (curated, size-aware server backends).
UAEF_EXTRA: str = _cfg["uaef"]["extra"]

#: The full pip requirement spec the Worker Lambda installs from the registry.
UAEF_PINNED_DEPENDENCY = f"uaef[{UAEF_EXTRA}]=={UAEF_VERSION}"

# --------------------------------------------------------------------------- #
# Shared resource naming (derived from config.yaml prefixes + deploy suffix).
# --------------------------------------------------------------------------- #

_naming = _cfg["naming"]

APP_NAME = f"{_naming['app_prefix']}-{_DEPLOY_SUFFIX}"
JOBS_TABLE_NAME = f"{_naming['jobs_table_prefix']}-{_DEPLOY_SUFFIX}"
PAYLOAD_BUCKET_NAME = f"{_naming['payload_bucket_prefix']}-{_DEPLOY_SUFFIX}"

# UAEF library persistence (reused as-is, NOT owned by this service).
UAEF_EXPERIMENT_TABLE_NAME = f"{_naming['experiment_table_prefix']}-{_DEPLOY_SUFFIX}"
UAEF_RESULTS_BUCKET_NAME = f"{_naming['results_bucket_prefix']}-{_DEPLOY_SUFFIX}"

# Cognito naming.
USER_POOL_NAME = f"{_naming['user_pool_prefix']}-{_DEPLOY_SUFFIX}"
USER_POOL_CLIENT_NAME = f"{_naming['user_pool_client_prefix']}-{_DEPLOY_SUFFIX}"

# Cognito Hosted-UI domain prefix (must be globally unique across AWS).
UI_DOMAIN_PREFIX = f"{_naming['ui_domain_prefix']}-{_DEPLOY_SUFFIX}"

# Note: UiStack hosts the deployed UI on CloudFront + S3, serving
# ``ui/app/dist`` (built by buildspec.yml). It takes no naming or source-path
# config beyond UI_DOMAIN_PREFIX above — the earlier Amplify-era constants
# (AMPLIFY_APP_NAME, AMPLIFY_BRANCH_NAME, UI_SOURCE_DIR,
# UI_API_BASE_URL_ENV_VAR) and their config.yaml keys were unused and removed.
# Whether the UI deploys at all is the ``deploy_ui`` CDK context flag, set from
# ``ui.deploy`` by scripts/cloud_deploy.sh.
