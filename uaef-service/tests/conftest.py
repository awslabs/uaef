# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the uaef-service job-state tests.

These tests exercise ``job_state`` against a mocked DynamoDB (via ``moto``).
The ``uaef-service`` directory is placed on ``sys.path`` so ``import job_state``
resolves to the module under test without importing the ``uaef`` package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
import pytest

# Make the uaef-service directory importable (so `import job_state` works) and
# ensure tests never accidentally talk to real AWS.
SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

TABLE_NAME = "uaef-service-jobs"


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch):
    """Set a deterministic region and dummy credentials for moto."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("JOBS_TABLE_NAME", TABLE_NAME)


@pytest.fixture
def jobs_table(_aws_env):
    """Create the mocked ``uaef-service-jobs`` table and reset job_state's cache.

    Yields the boto3 Table resource. job_state's cached ``_table`` is reset both
    before and after so it binds to the moto-backed resource.
    """
    from moto import mock_aws

    import job_state

    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        client.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": "jobId", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "jobId", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        client.get_waiter("table_exists").wait(TableName=TABLE_NAME)

        # Reset the module-level cached table so it picks up the mocked resource.
        job_state._table = None
        try:
            yield boto3.resource("dynamodb", region_name="us-east-1").Table(TABLE_NAME)
        finally:
            job_state._table = None
