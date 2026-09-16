#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate deploy_eks/helm/values.yaml from uaef-service/config.yaml.

This eliminates the need to maintain two config files. The user fills out
config.yaml once, and this script templates the Helm values from it.

Usage (from deploy_eks/):
    python3 generate_values.py

Or from uaef-service/:
    python3 ../deploy_eks/generate_values.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR.parent / "uaef-service" / "config.yaml"
VALUES_PATH = SCRIPT_DIR / "helm" / "values.yaml"


def main() -> int:
    if not CONFIG_PATH.exists():
        print(f"ERROR: config.yaml not found at {CONFIG_PATH}", file=sys.stderr)
        return 1

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    suffix = cfg.get("deploy_suffix", "")
    region = cfg.get("deploy_region", "us-east-1")
    auth = cfg.get("auth", {})
    uaef = cfg.get("uaef", {})
    naming = cfg.get("naming", {})
    eks = cfg.get("eks", {})

    # Derive resource names from naming prefixes + suffix
    jobs_table = f"{naming.get('jobs_table_prefix', 'uaef-service-jobs')}-{suffix}"
    payload_bucket = f"{naming.get('payload_bucket_prefix', 'uaef-service-payloads')}-{suffix}"
    experiment_table = f"{naming.get('experiment_table_prefix', 'uaef-experiments')}-{suffix}"
    results_bucket = f"{naming.get('results_bucket_prefix', 'uaef-results')}-{suffix}"

    # EKS-specific names with suffix
    ecr_repo_name = f"{eks.get('ecr_repo_prefix', 'uaef-service-eks')}-{suffix}"
    helm_release = f"{eks.get('helm_release_prefix', 'uaef-service')}-{suffix}"

    # Build the values dict
    values = {
        "replicaCount": eks.get("replica_count", 2),
        "cognito": {
            "userPoolId": auth.get("existing_user_pool_id", ""),
            "appClientId": auth.get("existing_app_client_id", ""),
            "region": region,
            "domain": auth.get("existing_cognito_domain", ""),
            "authMode": auth.get("mode", "existing"),
        },
        "aws": {
            "region": region,
            "jobsTableName": jobs_table,
            "payloadBucketName": payload_bucket,
            "experimentTableName": experiment_table,
            "resultsBucketName": results_bucket,
        },
        "serviceAccount": {
            "create": True,
            "name": "uaef-service",
            "annotations": {},
        },
        "uaef": {
            "expectedVersion": uaef.get("version", "0.2.0"),
        },
        "orchestrator": {
            "stateMachineArn": eks.get("state_machine_arn", ""),
            "workerFunctionName": eks.get("worker_function_name", ""),
        },
        "image": {
            "repository": eks.get("ecr_repository", ""),
            "tag": eks.get("image_tag", "latest"),
            "pullPolicy": "Always",
        },
        "service": {
            "type": "ClusterIP",
            "port": 8080,
        },
        "ingress": {
            "enabled": eks.get("ingress", {}).get("enabled", False),
            "className": eks.get("ingress", {}).get("class_name", "alb"),
            "annotations": {},
            "hosts": [
                {
                    "host": eks.get("ingress", {}).get("host", ""),
                    "paths": [{"path": "/", "pathType": "Prefix"}],
                }
            ],
        },
        "resources": {
            "requests": {
                "cpu": eks.get("resources", {}).get("cpu_request", "500m"),
                "memory": eks.get("resources", {}).get("memory_request", "1Gi"),
            },
            "limits": {
                "cpu": eks.get("resources", {}).get("cpu_limit", "2000m"),
                "memory": eks.get("resources", {}).get("memory_limit", "4Gi"),
            },
        },
        "livenessProbe": {
            "httpGet": {"path": "/health", "port": 8080},
            "initialDelaySeconds": 15,
            "periodSeconds": 30,
        },
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": 8080},
            "initialDelaySeconds": 10,
            "periodSeconds": 10,
        },
        "nodeSelector": {},
        "tolerations": [],
        "affinity": {},
    }

    # Add IRSA annotation if provided
    role_arn = eks.get("service_account_role_arn", "")
    if role_arn:
        values["serviceAccount"]["annotations"] = {
            "eks.amazonaws.com/role-arn": role_arn
        }

    # Write values.yaml
    VALUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VALUES_PATH, "w") as f:
        f.write("# Auto-generated from uaef-service/config.yaml by generate_eks_values.py\n")
        f.write("# Do not edit directly — modify config.yaml and re-run the deploy.\n\n")
        yaml.dump(values, f, default_flow_style=False, sort_keys=False)

    # Write a small metadata file so deploy.sh can read derived names
    meta = {"ecr_repo_name": ecr_repo_name, "helm_release": helm_release}
    meta_path = SCRIPT_DIR / "helm" / ".deploy_meta.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    print(f"Generated {VALUES_PATH}")
    print(f"  ECR repo name:    {ecr_repo_name}")
    print(f"  Helm release:     {helm_release}")
    print(f"  Jobs table:       {jobs_table}")
    print(f"  Payload bucket:   {payload_bucket}")
    print(f"  Experiment table: {experiment_table}")
    print(f"  Results bucket:   {results_bucket}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
