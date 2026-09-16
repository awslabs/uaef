#!/usr/bin/env bash
# Build the docs site and fail on broken links or unresolved API references.
#
# Why not `mkdocs build --strict`? Strict mode escalates *every* warning,
# including griffe's "No type or annotation for parameter ..." notices about
# missing annotations in src/uaef. Those are real gaps but they live in library
# code, not in the docs, and they would block every docs change until fixed.
# This script keeps the check that the docs can actually break — links — hard.
set -euo pipefail

LOG=$(mktemp -t uaef-mkdocs.XXXXXX)
trap 'rm -f "$LOG"' EXIT

uv run mkdocs build "$@" 2>&1 | tee "$LOG"

if grep -qE "contains a link|not found among documentation files|Could not find cross-reference|unrecognized relative link" "$LOG"; then
    echo >&2
    echo "ERROR: broken links or unresolved references (see warnings above)." >&2
    exit 1
fi

echo "Docs built with no link or reference problems."
