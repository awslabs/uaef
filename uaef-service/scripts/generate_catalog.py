#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the static metric catalog JSON at build time.

This script imports the uaef library's ``get_full_metric_catalog()`` and writes
the result to ``handlers/catalog.json``. The API Lambda loads this file directly
at startup, eliminating the need to synchronously invoke the Worker Lambda just
to fetch the (deploy-time-static) metric catalog.

Prereqs: the uaef wheel must be installed (``pip install ./wheels/uaef-*.whl``).
The script only needs the base ``uaef`` package — not the ``[server]`` extra —
because ``get_full_metric_catalog()`` gracefully omits integration groups whose
extras are not installed.

Usage (from uaef-service/):
    python3 scripts/generate_catalog.py

The output file is handlers/catalog.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Resolve paths relative to this script's location.
SCRIPT_DIR = Path(__file__).resolve().parent
SERVICE_DIR = SCRIPT_DIR.parent
OUTPUT_PATH = SERVICE_DIR / "handlers" / "catalog.json"


def main() -> int:
    try:
        from uaef.metrics import get_full_metric_catalog
    except ImportError as exc:
        print(
            f"ERROR: Could not import uaef.metrics: {exc}\n"
            "Install the uaef wheel first:\n"
            "    pip install ./wheels/uaef-*.whl",
            file=sys.stderr,
        )
        return 1

    print("Generating metric catalog...")
    catalog = get_full_metric_catalog()

    OUTPUT_PATH.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(catalog)} groups to {OUTPUT_PATH}")

    # Print summary.
    total_metrics = sum(len(v) for v in catalog.values())
    print(f"  Groups: {list(catalog.keys())}")
    print(f"  Total metrics: {total_metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
