#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the container image and deploy to EKS.
# Can be run repeatedly against any account/cluster.
#
# Usage:
#   ./deploy.sh                          # Build only
#   ./deploy.sh --push                   # Build + push to ECR
#   ./deploy.sh --push --helm-install    # Build + push + helm install
#
# Environment variables:
#   AWS_ACCOUNT    - AWS account ID (auto-detected if not set)
#   AWS_REGION     - AWS region (default: us-east-1)
#   ECR_REPO_NAME  - ECR repository name (default: uaef-service-eks)
#   IMAGE_TAG      - Docker image tag (default: latest)
#   HELM_RELEASE   - Helm release name (default: uaef-service)
#   HELM_NAMESPACE - K8s namespace (default: default)
#   EKS_CLUSTER    - EKS cluster name (required for --helm-install)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT="${AWS_ACCOUNT:-$(aws sts get-caller-identity --query Account --output text)}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
HELM_NAMESPACE="${HELM_NAMESPACE:-default}"

# Read derived names from generate_eks_values.py metadata (if available)
DEPLOY_META="${SCRIPT_DIR}/helm/.deploy_meta.yaml"
if [ -f "${DEPLOY_META}" ]; then
    ECR_REPO_NAME="${ECR_REPO_NAME:-$(python3 -c "import yaml; print(yaml.safe_load(open('${DEPLOY_META}'))['ecr_repo_name'])")}"
    HELM_RELEASE="${HELM_RELEASE:-$(python3 -c "import yaml; print(yaml.safe_load(open('${DEPLOY_META}'))['helm_release'])")}"
else
    ECR_REPO_NAME="${ECR_REPO_NAME:-uaef-service-eks}"
    HELM_RELEASE="${HELM_RELEASE:-uaef-service}"
fi

ECR_REPO="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

# Check source has been synced
if [ ! -d "${SCRIPT_DIR}/uaef-service/handlers" ]; then
    echo "ERROR: Handler source not found. Run ./sync-uaef.sh first."
    exit 1
fi

# Check values.yaml exists
if [ ! -f "${SCRIPT_DIR}/helm/values.yaml" ]; then
    echo "ERROR: helm/values.yaml not found."
    echo "  Copy the example and fill in your values:"
    echo "  cp helm/values.yaml.example helm/values.yaml"
    exit 1
fi

# Parse flags
DO_PUSH=false
DO_HELM=false
for arg in "$@"; do
    case "$arg" in
        --push) DO_PUSH=true ;;
        --helm-install) DO_HELM=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# Build + Push (multi-platform for both x86_64 and ARM/Graviton nodes)
# Read platforms from config.yaml (default: linux/amd64,linux/arm64)
PLATFORMS="${PLATFORMS:-$(python3 -c "
import yaml
with open('${SCRIPT_DIR}/../uaef-service/config.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('eks', {}).get('platforms', 'linux/amd64,linux/arm64'))
" 2>/dev/null || echo "linux/amd64,linux/arm64")}"

echo "=== Build platforms: ${PLATFORMS} ==="

# Ensure buildx builder exists
if ! docker buildx inspect uaef-builder >/dev/null 2>&1; then
    docker buildx create --name uaef-builder --use
else
    docker buildx use uaef-builder
fi

if [ "$DO_PUSH" = true ]; then
    echo "=== Ensuring ECR repository exists ==="
    aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
        aws ecr create-repository --repository-name "${ECR_REPO_NAME}" --region "${AWS_REGION}" >/dev/null

    aws ecr get-login-password --region "${AWS_REGION}" | \
        docker login --username AWS --password-stdin "${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

    docker buildx build \
        --platform "${PLATFORMS}" \
        -t "${ECR_REPO}:${IMAGE_TAG}" \
        --push .
    echo "=== Image pushed (platforms: ${PLATFORMS}) ==="
else
    # Build locally (single platform matching host) for testing
    docker buildx build --platform linux/amd64 -t "uaef-service-eks:${IMAGE_TAG}" --load .
    echo "=== Local build complete (amd64 only; use --push for multi-arch) ==="
fi

# Helm install/upgrade
if [ "$DO_HELM" = true ]; then
    if ! command -v helm &> /dev/null; then
        echo "=== Helm not found. Installing via Homebrew ==="
        if command -v brew &> /dev/null; then
            brew install helm
        else
            echo "ERROR: Helm not installed and Homebrew not available."
            echo "  Install Helm manually: https://helm.sh/docs/intro/install/"
            exit 1
        fi
    fi

    echo "=== Deploying via Helm: ${HELM_RELEASE} in namespace ${HELM_NAMESPACE} ==="

    EKS_CLUSTER="${EKS_CLUSTER:-}"
    if [ -z "$EKS_CLUSTER" ]; then
        echo "Enter EKS cluster name (or set EKS_CLUSTER env var):"
        printf "> "
        read -r EKS_CLUSTER
    fi
    if [ -z "$EKS_CLUSTER" ]; then
        echo "ERROR: EKS cluster name is required for --helm-install."
        exit 1
    fi

    echo "=== Configuring kubectl for cluster: ${EKS_CLUSTER} ==="
    aws eks update-kubeconfig --name "${EKS_CLUSTER}" --region "${AWS_REGION}"

    # Ensure IAM policy exists and is attached to the node role
    echo "=== Ensuring IAM permissions for EKS nodes ==="
    POLICY_NAME="uaef-service-eks-policy-${EKS_CLUSTER}"
    POLICY_ARN="arn:aws:iam::${AWS_ACCOUNT}:policy/${POLICY_NAME}"

    # Parse resource names from values.yaml (handles both quoted and unquoted YAML)
    JOBS_TABLE=$(grep 'jobsTableName' helm/values.yaml | awk '{print $2}' | tr -d '"' | tr -d "'")
    PAYLOAD_BUCKET=$(grep 'payloadBucketName' helm/values.yaml | awk '{print $2}' | tr -d '"' | tr -d "'")
    EXPERIMENT_TABLE=$(grep 'experimentTableName' helm/values.yaml | awk '{print $2}' | tr -d '"' | tr -d "'")
    RESULTS_BUCKET=$(grep 'resultsBucketName' helm/values.yaml | awk '{print $2}' | tr -d '"' | tr -d "'")

    if [[ -z "$JOBS_TABLE" || -z "$PAYLOAD_BUCKET" || -z "$EXPERIMENT_TABLE" || -z "$RESULTS_BUCKET" ]]; then
        echo "  ERROR: Could not parse resource names from helm/values.yaml"
        echo "    jobsTableName=$JOBS_TABLE payloadBucketName=$PAYLOAD_BUCKET"
        echo "    experimentTableName=$EXPERIMENT_TABLE resultsBucketName=$RESULTS_BUCKET"
        exit 1
    fi

    sed -e "s|JOBS_TABLE_NAME|${JOBS_TABLE}|g" \
        -e "s|UAEF_EXPERIMENT_TABLE_NAME|${EXPERIMENT_TABLE}|g" \
        -e "s|PAYLOAD_BUCKET_NAME|${PAYLOAD_BUCKET}|g" \
        -e "s|UAEF_RESULTS_BUCKET_NAME|${RESULTS_BUCKET}|g" \
        -e "s|AWS_ACCOUNT_ID|${AWS_ACCOUNT}|g" \
        iam-policy.json > /tmp/uaef-eks-policy.json

    # Create or update the IAM policy
    if ! aws iam get-policy --policy-arn "${POLICY_ARN}" >/dev/null 2>&1; then
        echo "  Creating IAM policy: ${POLICY_NAME}"
        aws iam create-policy \
            --policy-name "${POLICY_NAME}" \
            --policy-document file:///tmp/uaef-eks-policy.json >/dev/null
    else
        echo "  Updating IAM policy: ${POLICY_NAME}"
        aws iam create-policy-version \
            --policy-arn "${POLICY_ARN}" \
            --policy-document file:///tmp/uaef-eks-policy.json \
            --set-as-default >/dev/null
    fi
    rm -f /tmp/uaef-eks-policy.json

    # Find and attach to the node instance role
    NODEGROUP_NAME=$(aws eks list-nodegroups --cluster-name "${EKS_CLUSTER}" --query 'nodegroups[0]' --output text 2>/dev/null || echo "")

    if [ -n "$NODEGROUP_NAME" ] && [ "$NODEGROUP_NAME" != "None" ]; then
        NODE_ROLE=$(aws eks describe-nodegroup \
            --cluster-name "${EKS_CLUSTER}" \
            --nodegroup-name "${NODEGROUP_NAME}" \
            --query 'nodegroup.nodeRole' --output text | awk -F'/' '{print $NF}')

        if [ -n "$NODE_ROLE" ] && [ "$NODE_ROLE" != "None" ]; then
            echo "  Attaching policy to node role: ${NODE_ROLE}"
            aws iam attach-role-policy \
                --role-name "${NODE_ROLE}" \
                --policy-arn "${POLICY_ARN}" 2>/dev/null || true
        else
            echo "  WARNING: Could not determine node role from nodegroup ${NODEGROUP_NAME}."
            echo "  Attach ${POLICY_ARN} manually, or configure IRSA in values.yaml."
        fi
    else
        echo "  WARNING: No nodegroups found for cluster ${EKS_CLUSTER}."
        echo "  The cluster may still be provisioning. Attach ${POLICY_ARN} manually,"
        echo "  or configure IRSA (eks.service_account_role_arn in config.yaml)."
    fi

    helm upgrade --install "${HELM_RELEASE}" ./helm \
        --namespace "${HELM_NAMESPACE}" \
        -f ./helm/values.yaml \
        --set "image.repository=${ECR_REPO}" \
        --set "image.tag=${IMAGE_TAG}"

    # Force pods to pull the new image
    kubectl rollout restart deployment "${HELM_RELEASE}" -n "${HELM_NAMESPACE}"
    echo "=== Waiting for rollout to complete ==="
    kubectl rollout status deployment "${HELM_RELEASE}" -n "${HELM_NAMESPACE}" --timeout=120s

    echo "=== Helm deploy complete ==="
fi
