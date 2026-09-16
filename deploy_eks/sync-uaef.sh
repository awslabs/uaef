#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Sync UAEF handler source into deploy_eks/ for containerization.
# Run this once initially, then again when uaef-service handler code changes.
#
# Usage:
#   ./sync-uaef.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# The uaef-service directory is a sibling in the same repo
UAEF_SERVICE_DIR="${SCRIPT_DIR}/../uaef-service"

if [ ! -d "${UAEF_SERVICE_DIR}/handlers" ]; then
    echo "ERROR: Cannot find uaef-service at ${UAEF_SERVICE_DIR}"
    echo "  Expected to find a 'handlers/' directory there."
    exit 1
fi

echo "=== Syncing handler code from ${UAEF_SERVICE_DIR} ==="

# Clean previous copies
rm -rf "${SCRIPT_DIR}/uaef-service" "${SCRIPT_DIR}/wheels"

# Copy handler modules
mkdir -p "${SCRIPT_DIR}/uaef-service"
cp -r "${UAEF_SERVICE_DIR}/handlers" "${SCRIPT_DIR}/uaef-service/handlers"
cp "${UAEF_SERVICE_DIR}/job_state.py" "${SCRIPT_DIR}/uaef-service/"
cp "${UAEF_SERVICE_DIR}/schemas.py" "${SCRIPT_DIR}/uaef-service/"
cp "${UAEF_SERVICE_DIR}/validation.py" "${SCRIPT_DIR}/uaef-service/"

# Copy wheels (for registry-free install path)
if [ -d "${UAEF_SERVICE_DIR}/wheels" ]; then
    cp -r "${UAEF_SERVICE_DIR}/wheels" "${SCRIPT_DIR}/wheels"
else
    mkdir -p "${SCRIPT_DIR}/wheels"
fi

echo ""
echo "=== Source synced successfully ==="
echo "  uaef-service/handlers/"
echo "  uaef-service/job_state.py"
echo "  uaef-service/schemas.py"
echo "  uaef-service/validation.py"
echo "  wheels/"
echo ""
echo "Next: run ./deploy.sh --push --helm-install to build and deploy."
