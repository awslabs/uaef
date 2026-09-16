# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for owner-scoped experiment access (security review H-02).

Covers ``uaef.storage.dynamodb_s3.DynamoS3Storage``:

  * ``save_experiment(created_by=...)`` stamps the owner on a new experiment.
  * ``list_experiments(created_by=...)`` scopes results to that owner via the
    ``created_by`` GSI, never a full unscoped scan.
  * ``list_experiments()`` with no ``created_by`` (library-only callers with no
    per-caller identity concept) is unchanged: it still returns every
    experiment.
  * Experiments persisted before this fix (no ``created_by`` attribute) are
    never returned by an owner-scoped listing — the intended fail-closed
    behavior for pre-existing, ownerless rows.
  * ``created_by`` is preserved across an update that omits it, or that names
    a different caller — an experiment's owner cannot change after creation.

Exercised against a moto-mocked DynamoDB (with the ``created_by-index`` GSI,
matching ``uaef-service/infra/storage_stack.py``) + S3.
"""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from uaef.config import StorageConfig
from uaef.storage.dynamodb_s3 import DynamoS3Storage

TABLE_NAME = "test-uaef-experiments"
BUCKET_NAME = "test-uaef-results"
REGION = "us-east-1"


@pytest.fixture
def store(monkeypatch):
    """A DynamoS3Storage backed by a moto-mocked table (with the created_by GSI) + bucket."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")

    with mock_aws():
        ddb = boto3.resource("dynamodb", region_name=REGION)
        ddb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": "experiment_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "experiment_id", "AttributeType": "S"},
                {"AttributeName": "created_by", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
            GlobalSecondaryIndexes=[
                {
                    "IndexName": DynamoS3Storage.CREATED_BY_INDEX_NAME,
                    "KeySchema": [{"AttributeName": "created_by", "KeyType": "HASH"}],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
        )
        boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET_NAME)

        config = StorageConfig(
            dynamodb_table_name=TABLE_NAME,
            s3_bucket=BUCKET_NAME,
            s3_prefix="evaluations/",
            region=REGION,
        )
        yield DynamoS3Storage(config=config)


def _save(store, experiment_id, created_by=None, score=0.5):
    return store.save_experiment(
        experiment_id=experiment_id,
        experiment_name=f"exp {experiment_id}",
        evaluation_results=[{"overall_score": score, "dimension_results": []}],
        created_by=created_by,
    )


# --------------------------------------------------------------------------- #
# Owner scoping
# --------------------------------------------------------------------------- #


def test_list_experiments_scoped_to_owner_returns_only_own(store):
    _save(store, "exp-alice", created_by="alice-sub")
    _save(store, "exp-bob", created_by="bob-sub")

    alice_results = store.list_experiments(created_by="alice-sub")
    assert [r["experiment_id"] for r in alice_results] == ["exp-alice"]

    bob_results = store.list_experiments(created_by="bob-sub")
    assert [r["experiment_id"] for r in bob_results] == ["exp-bob"]


def test_list_experiments_scoped_to_owner_with_no_experiments_returns_empty(store):
    _save(store, "exp-alice", created_by="alice-sub")

    assert store.list_experiments(created_by="nobody-sub") == []


def test_list_experiments_unscoped_still_returns_everything(store):
    """Library-only callers with no per-caller identity concept: unchanged behavior."""
    _save(store, "exp-alice", created_by="alice-sub")
    _save(store, "exp-bob", created_by="bob-sub")
    _save(store, "exp-anonymous")  # no created_by at all

    all_results = store.list_experiments()
    assert {r["experiment_id"] for r in all_results} == {
        "exp-alice",
        "exp-bob",
        "exp-anonymous",
    }


def test_legacy_ownerless_experiment_never_returned_by_owner_scoped_list(store):
    """Fail-closed: an experiment with no created_by is inaccessible via owner-scoped listing."""
    _save(store, "exp-legacy")  # simulates data persisted before this fix

    for owner in ("alice-sub", "bob-sub", "nobody-sub"):
        results = store.list_experiments(created_by=owner)
        assert "exp-legacy" not in {r["experiment_id"] for r in results}

    # But it's still visible via the unscoped listing (library-only path).
    assert "exp-legacy" in {r["experiment_id"] for r in store.list_experiments()}


# --------------------------------------------------------------------------- #
# created_by immutability across updates
# --------------------------------------------------------------------------- #


def test_created_by_preserved_when_omitted_on_update(store):
    _save(store, "exp-alice", created_by="alice-sub", score=0.5)
    _save(store, "exp-alice", created_by=None, score=0.9)  # update omits created_by

    updated = store.get_experiment("exp-alice")
    assert updated["created_by"] == "alice-sub"


def test_created_by_cannot_be_overwritten_by_a_different_caller(store):
    _save(store, "exp-alice", created_by="alice-sub", score=0.5)
    _save(store, "exp-alice", created_by="mallory-sub", score=0.99)  # attempted takeover

    updated = store.get_experiment("exp-alice")
    assert updated["created_by"] == "alice-sub"


def test_created_at_and_created_by_both_preserved_across_multiple_updates(store):
    first = _save(store, "exp-alice", created_by="alice-sub")
    _save(store, "exp-alice", created_by="alice-sub", score=0.7)
    third = _save(store, "exp-alice", score=0.8)

    assert third["created_at"] == first["created_at"]
    assert third["created_by"] == "alice-sub"
