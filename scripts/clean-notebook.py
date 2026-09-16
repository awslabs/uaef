#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Git clean filter: strip outputs and execution counts from Jupyter notebooks.

Used as a git filter to automatically clean notebooks on `git add`.
Configure in .gitattributes + .git/config.
"""

import json
import sys


def clean_notebook(nb):
    """Remove outputs and execution counts from a notebook dict."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    # Reset kernel metadata that changes per-run
    metadata = nb.get("metadata", {})
    if "kernelspec" in metadata:
        pass  # keep kernelspec (useful info)
    if "language_info" in metadata:
        metadata["language_info"].pop("version", None)
    return nb


def main():
    try:
        nb = json.load(sys.stdin)
        cleaned = clean_notebook(nb)
        json.dump(cleaned, sys.stdout, indent=1, ensure_ascii=False)
        sys.stdout.write("\n")
    except json.JSONDecodeError:
        # Not valid JSON — pass through unchanged
        sys.stdin.seek(0)
        sys.stdout.write(sys.stdin.read())


if __name__ == "__main__":
    main()
