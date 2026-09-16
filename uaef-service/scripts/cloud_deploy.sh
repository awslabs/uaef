#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#
# cloud_deploy.sh — unified deploy script for the UAEF Service.
#
# Reads deploy_mode from config.yaml and dispatches to the appropriate flow:
#
#   deploy_mode: "with_ui"
#       Deploy UAEF with its built-in UI via CDK + CodeBuild.
#       (API Gateway + Lambda + S3/CloudFront + Cognito)
#
#   deploy_mode: "with_eks"
#       Containerize UAEF logic and deploy to EKS via Docker + Helm.
#       Connect your own external UI to the service endpoints.
#
# All configuration is read from config.yaml. Just edit it and run:
#     ./scripts/cloud_deploy.sh
#
# Optional env overrides:
#     AWS_REGION=us-east-1 ./scripts/cloud_deploy.sh

set -euo pipefail

# --------------------------------------------------------------------------- #
# Locate config.yaml and helpers
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${SERVICE_DIR}/config.yaml"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: config.yaml not found at ${CONFIG_FILE}"
  exit 1
fi

cfg_get_top() {
  local key="$1"
  python3 -c "
import yaml
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
v = cfg.get('${key}', '')
print(v if v else '')
"
}

cfg_get() {
  local section="$1" key="$2"
  python3 -c "
import yaml
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
v = cfg.get('${section}', {}).get('${key}', '')
print(v if v else '')
"
}

# --------------------------------------------------------------------------- #
# Package a source directory into a zip archive.
# Prefers the `zip` binary when available; otherwise falls back to Python's
# zipfile module so the script also works on hosts without `zip` installed
# (e.g. SageMaker Studio Linux) in addition to macOS terminals.
#   $1 = source directory whose contents are added at the archive root
#   $2 = output zip file path
# --------------------------------------------------------------------------- #
make_source_zip() {
  local src_dir="$1" out_zip="$2"
  if command -v zip >/dev/null 2>&1; then
    (
      cd "${src_dir}"
      zip -r -q "${out_zip}" . \
        -x '.venv/*' 'cdk.out/*' '__pycache__/*' '*/__pycache__/*' \
           '.pytest_cache/*' '.hypothesis/*' '*.pyc' 'outputs.json' '.git/*' \
           'ui/app/node_modules/*' 'ui/app/dist/*'
    )
  else
    SRC_DIR="${src_dir}" OUT_ZIP="${out_zip}" python3 - <<'PY'
import fnmatch
import os
import zipfile

src_dir = os.environ["SRC_DIR"]
out_zip = os.environ["OUT_ZIP"]

# Matched against the archive-relative path (POSIX separators), mirroring the
# `zip -x` excludes used when the zip binary is available.
EXCLUDE = [
    ".venv/*", "cdk.out/*", "__pycache__/*", "*/__pycache__/*",
    ".pytest_cache/*", ".hypothesis/*", "*.pyc", "outputs.json", ".git/*",
    "ui/app/node_modules/*", "ui/app/dist/*",
]


def excluded(rel):
    return any(fnmatch.fnmatch(rel, pat) for pat in EXCLUDE)


with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(src_dir):
        # Prune excluded directories in-place so we don't descend into them.
        dirs[:] = [
            d for d in dirs
            if not excluded(
                os.path.relpath(os.path.join(root, d), src_dir).replace(os.sep, "/") + "/"
            )
        ]
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, src_dir).replace(os.sep, "/")
            if not excluded(rel):
                zf.write(full, rel)
PY
  fi
}

# --------------------------------------------------------------------------- #
# Read common configuration
# --------------------------------------------------------------------------- #
DEPLOY_MODE="$(cfg_get_top "deploy_mode")"
DEPLOY_MODE="${DEPLOY_MODE:-with_ui}"

DEPLOY_SUFFIX="$(cfg_get_top "deploy_suffix")"
if [[ -z "${DEPLOY_SUFFIX}" ]]; then
  echo "ERROR: deploy_suffix is empty in ${CONFIG_FILE}"
  exit 1
fi

# The suffix is appended to every resource name, and two of those — the payload
# and results buckets — live in S3's global namespace. Deploying with the shipped
# placeholder fails ~4 minutes in, inside CodeBuild, as an opaque nested-stack
# error (AWS::EarlyValidation::ResourceExistenceCheck) that names no resource.
# Catch it here, in two seconds, with a message that says what to do.
case "${DEPLOY_SUFFIX}" in
  youralias-* | YOURALIAS-* | example-00 | EXAMPLE-00 | REPLACE-* | replace-*)
    echo "ERROR: deploy_suffix is still the placeholder '${DEPLOY_SUFFIX}' in ${CONFIG_FILE}"
    echo ""
    echo "Set it to your alias plus a number, for example jdoe-01."
    echo "It is appended to every resource name, two of which are S3 buckets that"
    echo "must be unique across all of AWS. The placeholder's bucket names are"
    echo "already taken, so the deploy cannot succeed with it."
    exit 1
    ;;
esac

# The suffix flows into S3 bucket names, which admit only lowercase letters,
# digits and hyphens. An uppercase or underscored suffix fails at resource
# creation with an equally opaque error, so reject it up front too.
if [[ ! "${DEPLOY_SUFFIX}" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]]; then
  echo "ERROR: deploy_suffix '${DEPLOY_SUFFIX}' is not usable in an S3 bucket name."
  echo "Use lowercase letters, digits and hyphens only, starting and ending"
  echo "with a letter or digit. For example: jdoe-01"
  exit 1
fi

DEPLOY_REGION="$(cfg_get_top "deploy_region")"
REGION="${AWS_REGION:-${DEPLOY_REGION:-us-east-1}}"

# Security review M-01: required CORS allow-list, no default.
CORS_ALLOWED_ORIGINS="$(cfg_get_top "cors_allowed_origins")"
ACK_INSECURE_CORS="$(cfg_get_top "acknowledge_insecure_cors")"

# Validated here, before the script performs ANY side effect (building the
# wheel, creating the source bucket, uploading source, touching CodeBuild), so
# an incomplete config costs nothing and the "nothing has been deployed"
# message below is unconditionally true.
#
# with_ui does NOT require this value: the only origin that matters is the UI's
# own CloudFront domain, which this script resolves automatically (from the
# stack outputs on a redeploy, or via the bootstrap two-pass deploy on a first
# run — see the with_ui branch below). Anything set here is treated as an
# ADDITIONAL origin, merged with the derived one, so a local dev server can be
# allowed alongside the deployed UI.
#
# with_eks deploys no CloudFront distribution and no API Gateway, so there is
# nothing to derive from and the payload bucket's CORS rules still need an
# origin. It stays required there.
if [[ -z "${CORS_ALLOWED_ORIGINS}" && "${DEPLOY_MODE}" == "with_eks" ]]; then
  echo "ERROR: cors_allowed_origins is not set in ${CONFIG_FILE} (security review M-01)."
  echo ""
  echo "with_eks brings its own UI, so the origin cannot be derived. Add it:"
  echo ""
  echo "    cors_allowed_origins: \"https://your-ui.example.com\""
  echo "    acknowledge_insecure_cors: false"
  echo ""
  echo "It is required even though API Gateway is not deployed in this mode,"
  echo "because the payload S3 bucket's CORS rules use it."
  echo ""
  echo "Nothing has been deployed."
  exit 1
fi

# --------------------------------------------------------------------------- #
# Preflight: are this suffix's S3 bucket names actually available?
# --------------------------------------------------------------------------- #
# StorageStack creates two buckets named from deploy_suffix, and S3 bucket names
# are global across all of AWS — so a suffix can be unusable because a stranger's
# account already owns the name, in a brand-new account with nothing deployed.
#
# CloudFormation reports that as a nested changeset stuck "Currently in FAILED"
# with an AWS::EarlyValidation::ResourceExistenceCheck hook failure that names no
# resource, roughly six minutes into CodeBuild, after the Docker image has been
# built and pushed. Nothing in the CodeBuild log says "bucket name taken".
#
# head-bucket distinguishes the cases: 404 means the name is free, 403 means it
# exists and belongs to someone else (or we cannot see it), success means we own
# it already. Only block on a first deploy — on a redeploy the buckets are ours
# and are expected to exist.
STACK_NAME_PREFLIGHT="UaefServiceStack-${DEPLOY_SUFFIX}"
if ! aws cloudformation describe-stacks --stack-name "${STACK_NAME_PREFLIGHT}" \
       --region "${REGION}" >/dev/null 2>&1; then
  PAYLOAD_PREFIX="$(cfg_get "naming" "payload_bucket_prefix")"
  RESULTS_PREFIX="$(cfg_get "naming" "results_bucket_prefix")"
  PAYLOAD_PREFIX="${PAYLOAD_PREFIX:-uaef-service-payloads}"
  RESULTS_PREFIX="${RESULTS_PREFIX:-uaef-results}"

  TAKEN_BUCKETS=()
  for bucket in "${PAYLOAD_PREFIX}-${DEPLOY_SUFFIX}" "${RESULTS_PREFIX}-${DEPLOY_SUFFIX}"; do
    HEAD_ERR="$(aws s3api head-bucket --bucket "${bucket}" --region "${REGION}" 2>&1 >/dev/null || true)"
    if [[ -z "${HEAD_ERR}" ]]; then
      # We can see it, so it is ours — retained from an earlier deploy of this
      # suffix. CloudFormation will still refuse to create it, so flag it, but
      # say something different because this one is recoverable by deleting it.
      TAKEN_BUCKETS+=("${bucket}  (exists in THIS account — left over from a previous deploy)")
    elif printf '%s' "${HEAD_ERR}" | grep -q "404\|Not Found"; then
      : # available
    elif printf '%s' "${HEAD_ERR}" | grep -q "403\|Forbidden"; then
      TAKEN_BUCKETS+=("${bucket}  (name already taken in another AWS account)")
    fi
    # Any other error (no credentials, network, region) is not a name problem;
    # stay quiet and let the deploy surface it.
  done

  if (( ${#TAKEN_BUCKETS[@]} > 0 )); then
    echo "ERROR: deploy_suffix '${DEPLOY_SUFFIX}' cannot be used — its S3 bucket"
    echo "name(s) are already taken:"
    echo ""
    for entry in "${TAKEN_BUCKETS[@]}"; do
      echo "    ${entry}"
    done
    echo ""
    echo "S3 bucket names are global across ALL of AWS, so this happens even in a"
    echo "brand-new account: someone else already owns the name. Deploying anyway"
    echo "fails ~6 minutes into CodeBuild with an opaque nested-stack error that"
    echo "names no resource, so it is caught here instead."
    echo ""
    echo "Fix: set a more distinctive deploy_suffix in ${CONFIG_FILE}, e.g."
    echo "    deploy_suffix: \"${DEPLOY_SUFFIX}-$(date +%m%d)\""
    echo "then re-run this script."
    echo ""
    echo "Nothing has been deployed."
    exit 1
  fi
fi

echo "============================================================"
echo "UAEF Service Deploy"
echo "============================================================"
echo "Config file:   ${CONFIG_FILE}"
echo "Deploy mode:   ${DEPLOY_MODE}"
echo "Deploy suffix: ${DEPLOY_SUFFIX}"
echo "Region:        ${REGION}"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------- #
# Auto-rebuild uaef wheel from source
# --------------------------------------------------------------------------- #
UAEF_SRC_DIR="${SERVICE_DIR}/.."
existing_wheel() { ls "${SERVICE_DIR}"/wheels/uaef-*.whl >/dev/null 2>&1; }

if [[ -f "${UAEF_SRC_DIR}/pyproject.toml" ]]; then
  # uv is this project's build/package tool. This used to call `uv build`
  # unconditionally, which breaks any environment without it: JupyterLab and
  # SageMaker images ship pip but generally not uv, and the script runs under
  # `set -e`, so the deploy died on a bare "uv: command not found" before doing
  # anything at all.
  #
  # Bootstrap uv rather than substituting a different tool. pip installs the uv
  # binary into its scripts directory, which is often not on PATH in a notebook
  # environment, so resolve that directory explicitly instead of assuming.
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing it with pip (one-time) ..."
    python3 -m pip install --quiet --disable-pip-version-check uv >/dev/null 2>&1 || true
    for candidate in \
      "$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))' 2>/dev/null)" \
      "${HOME}/.local/bin"; do
      if [[ -n "${candidate}" && -x "${candidate}/uv" ]]; then
        PATH="${candidate}:${PATH}"
        export PATH
        break
      fi
    done
    if command -v uv >/dev/null 2>&1; then
      echo "  uv $(uv --version 2>/dev/null | awk '{print $2}') ready"
    fi
  fi

  if command -v uv >/dev/null 2>&1; then
    echo "Building uaef wheel from source (uv) ..."
    # A build failure must not kill the deploy when a usable wheel is already
    # present; fall through to that instead.
    BUILD_OK="true"
    (cd "${UAEF_SRC_DIR}" && uv build --quiet) || BUILD_OK="false"

    if [[ "${BUILD_OK}" == "true" ]] && ls "${UAEF_SRC_DIR}"/dist/uaef-*-py3-none-any.whl >/dev/null 2>&1; then
      mkdir -p "${SERVICE_DIR}/wheels"
      rm -f "${SERVICE_DIR}"/wheels/uaef-*.whl
      cp "${UAEF_SRC_DIR}"/dist/uaef-*-py3-none-any.whl "${SERVICE_DIR}/wheels/"
      echo "  Copied fresh wheel to ${SERVICE_DIR}/wheels/"
    elif existing_wheel; then
      echo "  WARNING: uv build failed; using the wheel already in wheels/."
      echo "           Fine for a prebuilt release; rebuild if you changed src/uaef."
    else
      echo "ERROR: uv build failed and no wheel is present in wheels/."
      echo "Nothing has been deployed."
      exit 1
    fi
  elif existing_wheel; then
    # The distributed zip ships a prebuilt wheel, so a machine where uv cannot
    # be installed (no network, locked-down pip) still deploys.
    echo "uv unavailable and could not be installed — using the prebuilt wheel in wheels/."
    echo "  This is the expected path for the distributed workshop zip."
  else
    echo "ERROR: uv is not installed, could not be installed, and there is no"
    echo "prebuilt wheel in ${SERVICE_DIR}/wheels/."
    echo ""
    echo "Install uv and re-run:  python3 -m pip install uv"
    echo "Nothing has been deployed."
    exit 1
  fi
  echo ""
fi

# --------------------------------------------------------------------------- #
# Dispatch based on deploy_mode
# --------------------------------------------------------------------------- #
case "${DEPLOY_MODE}" in
  with_ui)
    # =================================================================== #
    # MODE: with_ui — CDK deploy via CodeBuild
    # =================================================================== #

    # Auth settings
    AUTH_MODE="$(cfg_get "auth" "mode")"
    AUTH_MODE="${AUTH_MODE:-new}"
    EXISTING_USER_POOL_ID="$(cfg_get "auth" "existing_user_pool_id")"
    EXISTING_USER_POOL_ARN="$(cfg_get "auth" "existing_user_pool_arn")"
    EXISTING_APP_CLIENT_ID="$(cfg_get "auth" "existing_app_client_id")"
    EXISTING_COGNITO_DOMAIN="$(cfg_get "auth" "existing_cognito_domain")"

    # UI settings
    DEPLOY_UI="$(cfg_get "ui" "deploy")"
    DEPLOY_UI="${DEPLOY_UI:-true}"

    if [[ "${AUTH_MODE}" == "existing" && -z "${EXISTING_USER_POOL_ID}" ]]; then
      echo "ERROR: auth.mode is 'existing' but auth.existing_user_pool_id is empty in ${CONFIG_FILE}"
      exit 1
    fi

    PROJECT_NAME="${PROJECT_NAME:-uaef-service-deploy-${DEPLOY_SUFFIX}}"
    ROLE_NAME="${ROLE_NAME:-uaef-service-deploy-codebuild-role}"
    STACK_NAME="${STACK_NAME:-UaefServiceStack-${DEPLOY_SUFFIX}}"
    ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
    SRC_BUCKET="uaef-service-deploy-src-${ACCOUNT_ID}-${REGION}"
    SRC_KEY="source-${DEPLOY_SUFFIX}.zip"
    ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

    echo "Auth mode:     ${AUTH_MODE}"
    echo "Deploy UI:     ${DEPLOY_UI}"
    echo "Stack:         ${STACK_NAME}"
    echo "Account:       ${ACCOUNT_ID}"
    if [[ "${AUTH_MODE}" == "existing" ]]; then
      echo "User Pool ID:  ${EXISTING_USER_POOL_ID}"
      [[ -n "${EXISTING_USER_POOL_ARN}" ]] && echo "User Pool ARN: ${EXISTING_USER_POOL_ARN}"
      [[ -n "${EXISTING_APP_CLIENT_ID}" ]] && echo "App Client ID: ${EXISTING_APP_CLIENT_ID}"
      [[ -n "${EXISTING_COGNITO_DOMAIN}" ]] && echo "Cognito Domain: ${EXISTING_COGNITO_DOMAIN}"
    fi
    echo ""

    # Sanity: wheel check
    if ! ls "${SERVICE_DIR}"/wheels/uaef-*.whl >/dev/null 2>&1; then
      echo "ERROR: no uaef wheel found in ${SERVICE_DIR}/wheels/."
      echo "Build it first: (cd .. && uv build && cp dist/uaef-*-py3-none-any.whl uaef-service/wheels/)"
      exit 1
    fi

    # Source bucket
    if ! aws s3api head-bucket --bucket "${SRC_BUCKET}" 2>/dev/null; then
      echo "Creating source bucket s3://${SRC_BUCKET} ..."
      if [ "${REGION}" = "us-east-1" ]; then
        aws s3api create-bucket --bucket "${SRC_BUCKET}" --region "${REGION}" >/dev/null
      else
        aws s3api create-bucket --bucket "${SRC_BUCKET}" --region "${REGION}" \
          --create-bucket-configuration "LocationConstraint=${REGION}" >/dev/null
      fi
    fi

    # IAM role for CodeBuild
    if ! aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
      echo "Creating CodeBuild service role ${ROLE_NAME} ..."
      aws iam create-role --role-name "${ROLE_NAME}" \
        --assume-role-policy-document '{
          "Version": "2012-10-17",
          "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "codebuild.amazonaws.com"},
            "Action": "sts:AssumeRole"
          }]
        }' >/dev/null
      aws iam attach-role-policy --role-name "${ROLE_NAME}" \
        --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
      echo "Waiting 15s for the new role to propagate ..."
      sleep 15
    fi

    # CodeBuild project
    ENV_JSON="type=LINUX_CONTAINER,image=aws/codebuild/standard:7.0,computeType=BUILD_GENERAL1_LARGE,privilegedMode=true"
    SRC_JSON="type=S3,location=${SRC_BUCKET}/${SRC_KEY}"

    if aws codebuild batch-get-projects --names "${PROJECT_NAME}" --region "${REGION}" \
         --query 'projects[0].name' --output text 2>/dev/null | grep -q "^${PROJECT_NAME}$"; then
      echo "Updating CodeBuild project ${PROJECT_NAME} ..."
      aws codebuild update-project --name "${PROJECT_NAME}" --region "${REGION}" \
        --source "${SRC_JSON}" --artifacts "type=NO_ARTIFACTS" \
        --environment "${ENV_JSON}" --service-role "${ROLE_ARN}" \
        --timeout-in-minutes 60 >/dev/null
    else
      echo "Creating CodeBuild project ${PROJECT_NAME} ..."
      aws codebuild create-project --name "${PROJECT_NAME}" --region "${REGION}" \
        --source "${SRC_JSON}" --artifacts "type=NO_ARTIFACTS" \
        --environment "${ENV_JSON}" --service-role "${ROLE_ARN}" \
        --timeout-in-minutes 60 >/dev/null
    fi

    # Package and upload source
    TMP_ZIP="$(mktemp "${TMPDIR:-/tmp}/uaef-src-XXXXXX").zip"
    echo "Packaging source from ${SERVICE_DIR} ..."
    make_source_zip "${SERVICE_DIR}" "${TMP_ZIP}"
    echo "Uploading ${SRC_KEY} to s3://${SRC_BUCKET}/${SRC_KEY} ..."
    aws s3 cp "${TMP_ZIP}" "s3://${SRC_BUCKET}/${SRC_KEY}" --region "${REGION}"
    rm -f "${TMP_ZIP}"

    # Build CDK context
    CDK_CONTEXT="-c deploy_ui=${DEPLOY_UI} -c auth_mode=${AUTH_MODE}"
    if [[ "${AUTH_MODE}" == "existing" ]]; then
      CDK_CONTEXT="${CDK_CONTEXT} -c existing_user_pool_id=${EXISTING_USER_POOL_ID}"
      [[ -n "${EXISTING_USER_POOL_ARN}" ]] && \
        CDK_CONTEXT="${CDK_CONTEXT} -c existing_user_pool_arn=${EXISTING_USER_POOL_ARN}"
      [[ -n "${EXISTING_APP_CLIENT_ID}" ]] && \
        CDK_CONTEXT="${CDK_CONTEXT} -c existing_app_client_id=${EXISTING_APP_CLIENT_ID}"
      [[ -n "${EXISTING_COGNITO_DOMAIN}" ]] && \
        CDK_CONTEXT="${CDK_CONTEXT} -c existing_cognito_domain=${EXISTING_COGNITO_DOMAIN}"
    fi

    # --------------------------------------------------------------------- #
    # CORS origin (single-call deploy — no bootstrap, no second pass)
    # --------------------------------------------------------------------- #
    # The deployed UI's CloudFront origin is now resolved INSIDE the CDK app:
    # UiStack hands it to the API and payload-bucket CORS rules as an ordinary
    # CloudFormation reference (see infra/cors.py, infra/app.py). So this script
    # no longer deploys once with a wildcard origin to learn the CloudFront
    # domain and then redeploys with the real one — a single deploy locks CORS
    # to the real origin with no wildcard window.
    #
    # All this script does is forward any EXTRA origins from config.yaml (e.g. a
    # localhost dev server) as context; the UI's own origin is added in-stack.
    #
    #   * ui.deploy=true  — extra origins optional; the UI origin is derived.
    #   * ui.deploy=false — no UI is deployed, so no origin can be derived and
    #                       cors_allowed_origins MUST name the calling client.
    #
    # ui.deploy comes from YAML via python as "True"/"true"; lowercase with tr
    # rather than ${VAR,,} (bash 4+, a parse error on macOS's bash 3.2).
    DEPLOY_UI_LC="$(printf '%s' "${DEPLOY_UI}" | tr '[:upper:]' '[:lower:]')"

    EXTRA_ORIGINS="${CORS_ALLOWED_ORIGINS}"
    if [[ "${DEPLOY_UI_LC}" == "true" ]]; then
      # The derived UI origin is authoritative, so a lone wildcard in config is
      # meaningless noise — drop it rather than forwarding an insecure origin.
      [[ "${EXTRA_ORIGINS}" == "*" ]] && EXTRA_ORIGINS=""
    elif [[ -z "${EXTRA_ORIGINS}" ]]; then
      echo "ERROR: ui.deploy is false and cors_allowed_origins is empty in ${CONFIG_FILE}."
      echo ""
      echo "With no UI deployed there is no CloudFront origin to derive, so set the"
      echo "origin of whatever client calls this API:"
      echo ""
      echo "    cors_allowed_origins: \"https://your-client.example.com\""
      echo ""
      echo "Nothing has been deployed."
      exit 1
    fi

    # Forward extra/explicit origins only when set; the UI origin is derived
    # in-stack and needs no flag. i_acknowledge_insecure_cors is forwarded for
    # the ui.deploy=false explicit-wildcard case only (with_ui never sets it).
    if [[ -n "${EXTRA_ORIGINS}" ]]; then
      CDK_CONTEXT="${CDK_CONTEXT} -c api_cors_allowed_origins=${EXTRA_ORIGINS}"
      if [[ "${ACK_INSECURE_CORS}" == "true" ]]; then
        CDK_CONTEXT="${CDK_CONTEXT} -c i_acknowledge_insecure_cors=true"
      fi
    fi

    ENV_OVERRIDES="[\
{\"name\":\"CDK_CONTEXT\",\"value\":\"${CDK_CONTEXT}\",\"type\":\"PLAINTEXT\"},\
{\"name\":\"DEPLOY_SUFFIX\",\"value\":\"${DEPLOY_SUFFIX}\",\"type\":\"PLAINTEXT\"}]"

    # Start CodeBuild
    BUILD_ID="$(aws codebuild start-build --project-name "${PROJECT_NAME}" \
      --region "${REGION}" \
      --environment-variables-override "${ENV_OVERRIDES}" \
      --query 'build.id' --output text)"
    echo
    echo "Started build: ${BUILD_ID}"
    echo "Console: https://${REGION}.console.aws.amazon.com/codesuite/codebuild/${ACCOUNT_ID}/projects/${PROJECT_NAME}/build/${BUILD_ID//:/%3A}"
    echo

    echo "Streaming logs (Ctrl-C to detach; build continues) ..."
    sleep 8
    aws logs tail "/aws/codebuild/${PROJECT_NAME}" --follow --since 1m --region "${REGION}" 2>/dev/null &
    TAIL_PID=$!

    while true; do
      STATUS="$(aws codebuild batch-get-builds --ids "${BUILD_ID}" --region "${REGION}" \
        --query 'builds[0].buildStatus' --output text)"
      case "${STATUS}" in
        IN_PROGRESS) sleep 10 ;;
        *) break ;;
      esac
    done
    # CloudWatch Logs delivery lags the build by a few seconds. Killing the tail
    # the instant buildStatus flips drops exactly the lines that say WHY a build
    # failed — the last thing seen is a mid-progress message, then "FAILED" with
    # no cause. Let the tail drain first.
    sleep 12
    kill "${TAIL_PID}" 2>/dev/null || true

    echo
    echo "Build finished with status: ${STATUS}"

    if [ "${STATUS}" = "SUCCEEDED" ]; then
      echo
      echo "==== ${STACK_NAME} Outputs ===="
      aws cloudformation describe-stacks --stack-name "${STACK_NAME}" --region "${REGION}" \
        --query 'Stacks[0].Outputs[].{Key:OutputKey,Value:OutputValue}' --output table

      # ------------------------------------------------------------------- #
      # Tell the deployer how to log in
      # ------------------------------------------------------------------- #
      # auth_stack.py sets self_sign_up_enabled=False and a new pool is created
      # empty, so a successful deploy still leaves nobody able to sign in. The
      # outputs table above lists UserPoolId but not what to do with it, which
      # reads as "the UI is broken" rather than "create an account". Print the
      # exact commands, with the pool ID already filled in, and skip them if a
      # user already exists (a redeploy keeps the same pool and its users).
      STACK_OUTPUTS_JSON="$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" --region "${REGION}" \
        --query 'Stacks[0].Outputs' --output json 2>/dev/null || echo '[]')"
      read_output() {
        printf '%s' "${STACK_OUTPUTS_JSON}" | python3 -c "
import json,sys
key=sys.argv[1]
try:
    data=json.load(sys.stdin)
except Exception:
    data=[]
print(next((o.get('OutputValue','') for o in data if o.get('OutputKey')==key), ''))
" "$1" 2>/dev/null || true
      }
      POOL_ID="$(read_output UserPoolId)"
      UI_URL_OUT="$(read_output UiUrl)"

      if [[ -n "${POOL_ID}" ]]; then
        USER_COUNT="$(aws cognito-idp list-users --user-pool-id "${POOL_ID}" \
          --region "${REGION}" --max-results 1 \
          --query 'length(Users)' --output text 2>/dev/null || echo "unknown")"

        echo
        echo "============================================================"
        if [[ "${USER_COUNT}" == "0" ]]; then
          echo "ONE MORE STEP — create your login"
        else
          echo "Logging in"
        fi
        echo "============================================================"
        [[ -n "${UI_URL_OUT}" ]] && echo "UI:           ${UI_URL_OUT}"
        echo "User pool:    ${POOL_ID}"
        echo

        if [[ "${USER_COUNT}" == "0" ]]; then
          echo "The pool is empty and self sign-up is disabled, so create a user"
          echo "before opening the UI. Copy and paste, changing the username and"
          echo "password if you like (password: 12+ chars, upper, lower, digit, symbol):"
        else
          echo "This pool already has ${USER_COUNT} user(s) — reuse your existing login."
          echo "To add another:"
        fi
        echo
        echo "  aws cognito-idp admin-create-user \\"
        echo "    --user-pool-id ${POOL_ID} --region ${REGION} \\"
        echo "    --username workshop --message-action SUPPRESS \\"
        echo "    --user-attributes Name=email,Value=workshop@example.com Name=email_verified,Value=true"
        echo
        echo "  aws cognito-idp admin-set-user-password \\"
        echo "    --user-pool-id ${POOL_ID} --region ${REGION} \\"
        echo "    --username workshop --password 'Workshop!2026aef' --permanent"
        echo
        echo "Then sign in at the UI above with that username and password."
        echo "============================================================"
      fi
    else
      echo "Deploy did not succeed. Open the Console link above for the full log."
      exit 1
    fi
    ;;

  with_eks)
    # =================================================================== #
    # MODE: with_eks — Docker build + Helm deploy to EKS
    # =================================================================== #
    DEPLOY_EKS_DIR="${SERVICE_DIR}/../deploy_eks"

    if [[ ! -d "${DEPLOY_EKS_DIR}" ]]; then
      echo "ERROR: deploy_eks/ directory not found at ${DEPLOY_EKS_DIR}"
      exit 1
    fi

    # --------------------------------------------------------------------- #
    # Preflight: required local tooling
    # --------------------------------------------------------------------- #
    # This mode deploys in stages: CDK (storage/auth/worker/orchestrator) in
    # step 2, then `docker build`/`docker push` + `helm upgrade --install` in
    # step 3, then `kubectl get svc` in step 4. Without this check a missing
    # docker/helm/kubectl is only discovered *after* the CDK deploy has
    # already applied, leaving a half-deployed service that needs manual
    # reconciliation. Verify everything the later steps need up front, before
    # any AWS-side change is made.
    MISSING_TOOLS=()
    for tool in docker helm kubectl; do
      command -v "${tool}" >/dev/null 2>&1 || MISSING_TOOLS+=("${tool}")
    done

    if (( ${#MISSING_TOOLS[@]} > 0 )); then
      echo "ERROR: deploy_mode 'with_eks' requires local tooling that is missing:"
      for tool in "${MISSING_TOOLS[@]}"; do
        case "${tool}" in
          docker)  echo "  - docker  (build/push the service image)   install: https://docs.docker.com/get-docker/" ;;
          helm)    echo "  - helm    (install the Helm release)       install: brew install helm" ;;
          kubectl) echo "  - kubectl (read back the service endpoint) install: brew install kubectl" ;;
        esac
      done
      echo ""
      echo "Install the above and re-run. Nothing has been deployed."
      exit 1
    fi

    # A docker CLI with no reachable daemon fails the same way, just later and
    # more cryptically, so check reachability rather than just the binary.
    if ! docker info >/dev/null 2>&1; then
      echo "ERROR: the docker CLI is installed but no Docker daemon is reachable."
      echo "Start Docker (e.g. Docker Desktop, or 'colima start') and re-run."
      echo "Nothing has been deployed."
      exit 1
    fi

    # Validate or create EKS cluster
    EKS_CLUSTER="$(cfg_get "eks" "cluster_name")"
    CREATE_CLUSTER="$(cfg_get "eks" "create_cluster")"
    CREATE_CLUSTER="${CREATE_CLUSTER:-false}"

    # If no cluster name specified, derive from deploy suffix
    if [[ -z "${EKS_CLUSTER}" ]]; then
      EKS_CLUSTER="uaef-${DEPLOY_SUFFIX}"
      echo "No eks.cluster_name set — using derived name: ${EKS_CLUSTER}"
    fi
    export EKS_CLUSTER

    # Check if cluster exists
    if ! aws eks describe-cluster --name "${EKS_CLUSTER}" --region "${REGION}" >/dev/null 2>&1; then
      if [[ "${CREATE_CLUSTER}" == "true" || "${CREATE_CLUSTER}" == "True" ]]; then
        NODE_TYPE="$(cfg_get "eks" "node_type")"
        NODE_TYPE="${NODE_TYPE:-t3.medium}"
        NODE_COUNT="$(cfg_get "eks" "node_count")"
        NODE_COUNT="${NODE_COUNT:-2}"

        echo "EKS cluster '${EKS_CLUSTER}' not found. Creating..."
        echo "  Node type:  ${NODE_TYPE}"
        echo "  Node count: ${NODE_COUNT}"
        echo ""

        if ! command -v eksctl &> /dev/null; then
          echo "ERROR: eksctl not found. Install it to create clusters:"
          echo "  brew install eksctl"
          exit 1
        fi

        eksctl create cluster \
          --name "${EKS_CLUSTER}" \
          --region "${REGION}" \
          --nodes "${NODE_COUNT}" \
          --node-type "${NODE_TYPE}" \
          --managed

        echo ""
        echo "Cluster '${EKS_CLUSTER}' created successfully."

        # Write cluster name back to config.yaml
        python3 -c "
import yaml
config_path = '${CONFIG_FILE}'
with open(config_path) as f:
    cfg = yaml.safe_load(f)
if 'eks' not in cfg:
    cfg['eks'] = {}
cfg['eks']['cluster_name'] = '${EKS_CLUSTER}'
with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
        echo "  Updated config.yaml with cluster_name: ${EKS_CLUSTER}"
      else
        echo "ERROR: EKS cluster '${EKS_CLUSTER}' does not exist."
        echo "  Either create it manually, or set eks.create_cluster: true in config.yaml"
        exit 1
      fi
    fi

    echo "EKS cluster:   ${EKS_CLUSTER}"
    echo "Dispatching to EKS deployment flow..."
    echo ""

    # --------------------------------------------------------------------- #
    # Phase 1: Provision backend resources (DynamoDB, S3, Cognito) via CDK
    # --------------------------------------------------------------------- #
    echo "=== Phase 1: Provisioning backend resources via CDK (CodeBuild) ==="
    echo "  (DynamoDB tables, S3 buckets, Cognito, Worker Lambda, Step Functions)"
    echo "  (Skipped: API Gateway + UI — EKS serves the HTTP API)"
    echo ""

    AUTH_MODE="$(cfg_get "auth" "mode")"
    AUTH_MODE="${AUTH_MODE:-new}"
    EXISTING_USER_POOL_ID="$(cfg_get "auth" "existing_user_pool_id")"
    EXISTING_USER_POOL_ARN="$(cfg_get "auth" "existing_user_pool_arn")"
    EXISTING_APP_CLIENT_ID="$(cfg_get "auth" "existing_app_client_id")"
    EXISTING_COGNITO_DOMAIN="$(cfg_get "auth" "existing_cognito_domain")"

    PROJECT_NAME="${PROJECT_NAME:-uaef-service-deploy-${DEPLOY_SUFFIX}}"
    ROLE_NAME="${ROLE_NAME:-uaef-service-deploy-codebuild-role}"
    STACK_NAME="${STACK_NAME:-UaefServiceStack-${DEPLOY_SUFFIX}}"
    ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
    SRC_BUCKET="uaef-service-deploy-src-${ACCOUNT_ID}-${REGION}"
    SRC_KEY="source-${DEPLOY_SUFFIX}.zip"
    ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

    # Sanity: wheel check
    if ! ls "${SERVICE_DIR}"/wheels/uaef-*.whl >/dev/null 2>&1; then
      echo "ERROR: no uaef wheel found in ${SERVICE_DIR}/wheels/."
      echo "Build it first: (cd .. && uv build && cp dist/uaef-*-py3-none-any.whl uaef-service/wheels/)"
      exit 1
    fi

    # Source bucket
    if ! aws s3api head-bucket --bucket "${SRC_BUCKET}" 2>/dev/null; then
      echo "Creating source bucket s3://${SRC_BUCKET} ..."
      if [ "${REGION}" = "us-east-1" ]; then
        aws s3api create-bucket --bucket "${SRC_BUCKET}" --region "${REGION}" >/dev/null
      else
        aws s3api create-bucket --bucket "${SRC_BUCKET}" --region "${REGION}" \
          --create-bucket-configuration "LocationConstraint=${REGION}" >/dev/null
      fi
    fi

    # IAM role for CodeBuild
    if ! aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
      echo "Creating CodeBuild service role ${ROLE_NAME} ..."
      aws iam create-role --role-name "${ROLE_NAME}" \
        --assume-role-policy-document '{
          "Version": "2012-10-17",
          "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "codebuild.amazonaws.com"},
            "Action": "sts:AssumeRole"
          }]
        }' >/dev/null
      aws iam attach-role-policy --role-name "${ROLE_NAME}" \
        --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
      echo "Waiting 15s for the new role to propagate ..."
      sleep 15
    fi

    # CodeBuild project
    ENV_JSON="type=LINUX_CONTAINER,image=aws/codebuild/standard:7.0,computeType=BUILD_GENERAL1_LARGE,privilegedMode=true"
    SRC_JSON="type=S3,location=${SRC_BUCKET}/${SRC_KEY}"

    if aws codebuild batch-get-projects --names "${PROJECT_NAME}" --region "${REGION}" \
         --query 'projects[0].name' --output text 2>/dev/null | grep -q "^${PROJECT_NAME}$"; then
      aws codebuild update-project --name "${PROJECT_NAME}" --region "${REGION}" \
        --source "${SRC_JSON}" --artifacts "type=NO_ARTIFACTS" \
        --environment "${ENV_JSON}" --service-role "${ROLE_ARN}" \
        --timeout-in-minutes 60 >/dev/null
    else
      echo "Creating CodeBuild project ${PROJECT_NAME} ..."
      aws codebuild create-project --name "${PROJECT_NAME}" --region "${REGION}" \
        --source "${SRC_JSON}" --artifacts "type=NO_ARTIFACTS" \
        --environment "${ENV_JSON}" --service-role "${ROLE_ARN}" \
        --timeout-in-minutes 60 >/dev/null
    fi

    # Package and upload source
    TMP_ZIP="$(mktemp "${TMPDIR:-/tmp}/uaef-src-XXXXXX").zip"
    echo "Packaging source ..."
    make_source_zip "${SERVICE_DIR}" "${TMP_ZIP}"
    aws s3 cp "${TMP_ZIP}" "s3://${SRC_BUCKET}/${SRC_KEY}" --region "${REGION}" --quiet
    rm -f "${TMP_ZIP}"

    # CDK context: deploy_ui=false, deploy_api=false (skip API Gateway;
    # Worker + Orchestrator + Storage + Auth are deployed). EKS serves the
    # HTTP API itself but delegates evaluation to Step Functions + Worker Lambda.
    CDK_CONTEXT="-c deploy_ui=false -c deploy_api=false -c auth_mode=${AUTH_MODE}"
    if [[ "${AUTH_MODE}" == "existing" ]]; then
      CDK_CONTEXT="${CDK_CONTEXT} -c existing_user_pool_id=${EXISTING_USER_POOL_ID}"
      [[ -n "${EXISTING_USER_POOL_ARN}" ]] && \
        CDK_CONTEXT="${CDK_CONTEXT} -c existing_user_pool_arn=${EXISTING_USER_POOL_ARN}"
      [[ -n "${EXISTING_APP_CLIENT_ID}" ]] && \
        CDK_CONTEXT="${CDK_CONTEXT} -c existing_app_client_id=${EXISTING_APP_CLIENT_ID}"
      [[ -n "${EXISTING_COGNITO_DOMAIN}" ]] && \
        CDK_CONTEXT="${CDK_CONTEXT} -c existing_cognito_domain=${EXISTING_COGNITO_DOMAIN}"
    fi

    # Security review M-01: StorageStack (payload bucket CORS) still requires
    # this context even in EKS mode, where ApiStack itself is skipped.
    # Non-empty is already enforced up front (see the shared check after the
    # config read).
    CDK_CONTEXT="${CDK_CONTEXT} -c api_cors_allowed_origins=${CORS_ALLOWED_ORIGINS}"
    if [[ "${ACK_INSECURE_CORS}" == "true" ]]; then
      CDK_CONTEXT="${CDK_CONTEXT} -c i_acknowledge_insecure_cors=true"
    fi

    ENV_OVERRIDES="[{\"name\":\"CDK_CONTEXT\",\"value\":\"${CDK_CONTEXT}\",\"type\":\"PLAINTEXT\"},{\"name\":\"DEPLOY_SUFFIX\",\"value\":\"${DEPLOY_SUFFIX}\",\"type\":\"PLAINTEXT\"}]"

    # Start CodeBuild
    BUILD_ID="$(aws codebuild start-build --project-name "${PROJECT_NAME}" \
      --region "${REGION}" \
      --environment-variables-override "${ENV_OVERRIDES}" \
      --query 'build.id' --output text)"
    echo "Started CDK build: ${BUILD_ID}"
    echo "Console: https://${REGION}.console.aws.amazon.com/codesuite/codebuild/${ACCOUNT_ID}/projects/${PROJECT_NAME}/build/${BUILD_ID//:/%3A}"

    # Poll until done
    sleep 8
    aws logs tail "/aws/codebuild/${PROJECT_NAME}" --follow --since 1m --region "${REGION}" 2>/dev/null &
    TAIL_PID=$!

    while true; do
      STATUS="$(aws codebuild batch-get-builds --ids "${BUILD_ID}" --region "${REGION}" \
        --query 'builds[0].buildStatus' --output text)"
      case "${STATUS}" in
        IN_PROGRESS) sleep 10 ;;
        *) break ;;
      esac
    done
    # CloudWatch Logs delivery lags the build by a few seconds. Killing the tail
    # the instant buildStatus flips drops exactly the lines that say WHY a build
    # failed — the last thing seen is a mid-progress message, then "FAILED" with
    # no cause. Let the tail drain first.
    sleep 12
    kill "${TAIL_PID}" 2>/dev/null || true

    echo ""
    echo "CDK build finished with status: ${STATUS}"
    if [ "${STATUS}" != "SUCCEEDED" ]; then
      echo "Backend resource provisioning failed. Check the CodeBuild logs."
      exit 1
    fi

    # Read CDK outputs (StateMachineArn, WorkerFunctionName) and write to config.yaml
    # so generate_eks_values.py can include them in Helm values.
    echo "Reading CDK stack outputs..."
    STATE_MACHINE_ARN="$(aws cloudformation describe-stacks --stack-name "${STACK_NAME}" \
      --region "${REGION}" --query 'Stacks[0].Outputs[?OutputKey==`StateMachineArn`].OutputValue' \
      --output text 2>/dev/null || echo "")"
    WORKER_FN_NAME="$(aws cloudformation describe-stacks --stack-name "${STACK_NAME}" \
      --region "${REGION}" --query 'Stacks[0].Outputs[?OutputKey==`WorkerFunctionName`].OutputValue' \
      --output text 2>/dev/null || echo "")"

    if [[ -n "${STATE_MACHINE_ARN}" ]]; then
      echo "  State machine ARN: ${STATE_MACHINE_ARN}"
      echo "  Worker function:   ${WORKER_FN_NAME}"
      python3 -c "
import yaml
config_path = '${CONFIG_FILE}'
with open(config_path) as f:
    cfg = yaml.safe_load(f)
if 'eks' not in cfg:
    cfg['eks'] = {}
cfg['eks']['state_machine_arn'] = '${STATE_MACHINE_ARN}'
cfg['eks']['worker_function_name'] = '${WORKER_FN_NAME}'
with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    else
      echo "  (No StateMachineArn output — evaluation will run in-process)"
    fi
    echo ""

    # --------------------------------------------------------------------- #
    # Phase 2: Deploy container to EKS
    # --------------------------------------------------------------------- #
    echo "=== Phase 2: Deploying container to EKS ==="

    # Step 1: Generate helm/values.yaml from config.yaml
    echo "=== Step 1: Generating helm/values.yaml from config.yaml ==="
    python3 "${DEPLOY_EKS_DIR}/generate_eks_values.py"
    echo ""

    # Step 2: Sync handler source
    echo "=== Step 2: Syncing handler source ==="
    (cd "${DEPLOY_EKS_DIR}" && bash sync-uaef.sh)
    echo ""

    # Step 3: Build, push, and helm install
    echo "=== Step 3: Building, pushing, and deploying ==="
    export AWS_REGION="${REGION}"
    (cd "${DEPLOY_EKS_DIR}" && bash deploy.sh --push --helm-install)

    # Step 4: Auto-populate EKS outputs back into config.yaml
    echo ""
    echo "=== Step 4: Updating config.yaml with deployment outputs ==="
    ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
    ECR_REPO_PREFIX="$(cfg_get "eks" "ecr_repo_prefix")"
    ECR_REPO_PREFIX="${ECR_REPO_PREFIX:-uaef-service-eks}"
    ECR_REPO_NAME="${ECR_REPO_PREFIX}-${DEPLOY_SUFFIX}"
    ECR_REPOSITORY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"

    # Get service endpoint (LoadBalancer or ClusterIP)
    HELM_RELEASE_PREFIX="$(cfg_get "eks" "helm_release_prefix")"
    HELM_RELEASE_PREFIX="${HELM_RELEASE_PREFIX:-uaef-service}"
    HELM_RELEASE="${HELM_RELEASE_PREFIX}-${DEPLOY_SUFFIX}"
    HELM_NAMESPACE="$(cfg_get "eks" "namespace")"
    HELM_NAMESPACE="${HELM_NAMESPACE:-default}"
    SERVICE_ENDPOINT="$(kubectl get svc "${HELM_RELEASE}" -n "${HELM_NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")"
    if [[ -z "${SERVICE_ENDPOINT}" ]]; then
      SERVICE_ENDPOINT="$(kubectl get svc "${HELM_RELEASE}" -n "${HELM_NAMESPACE}" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")"
      if [[ -n "${SERVICE_ENDPOINT}" ]]; then
        SVC_PORT="$(kubectl get svc "${HELM_RELEASE}" -n "${HELM_NAMESPACE}" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "8080")"
        SERVICE_ENDPOINT="${SERVICE_ENDPOINT}:${SVC_PORT}"
      fi
    fi

    # Write outputs back to config.yaml
    python3 -c "
import yaml

config_path = '${CONFIG_FILE}'
with open(config_path) as f:
    cfg = yaml.safe_load(f)

if 'eks' not in cfg:
    cfg['eks'] = {}

cfg['eks']['ecr_repository'] = '${ECR_REPOSITORY}'
cfg['eks']['service_endpoint'] = '${SERVICE_ENDPOINT}'

with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    echo "  ECR repository:   ${ECR_REPOSITORY}"
    echo "  Service endpoint: ${SERVICE_ENDPOINT}"
    echo ""
    echo "=== EKS deployment complete ==="
    ;;

  *)
    echo "ERROR: Unknown deploy_mode '${DEPLOY_MODE}' in ${CONFIG_FILE}"
    echo "  Valid options: 'with_ui' or 'with_eks'"
    exit 1
    ;;
esac
