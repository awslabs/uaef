# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for UAEF utilities."""

import os
from pathlib import Path

# Resolve the repo root (3 levels up from this file: utils -> uaef -> src -> root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Default output directory for batch evaluation metric results
BATCH_EVAL_OUTPUT_DIR = str(_REPO_ROOT / "output" / "evaluation-results")

# Default output directory for reports
REPORTS_OUTPUT_DIR = str(_REPO_ROOT / "output" / "reports")
