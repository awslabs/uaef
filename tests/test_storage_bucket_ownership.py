# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3 bucket naming and ownership checks for DynamoS3Storage.

Bucket names are global, so the storage backend must never fall back to a
fixed name that another account could own, and must pin every S3 call to the
caller's own account.
"""

from unittest.mock import MagicMock

import boto3
import pytest
from botocore.exceptions import ClientError

pytest.importorskip("moto")
from moto import mock_aws  # noqa: E402

from uaef.config import StorageConfig  # noqa: E402
from uaef.storage.dynamodb_s3 import DynamoS3Storage  # noqa: E402

REGION = "us-east-1"
MOTO_ACCOUNT = "123456789012"


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.delenv("UAEF_S3_BUCKET", raising=False)
    with mock_aws():
        yield


def _config(**kwargs):
    return StorageConfig(dynamodb_table_name="t", region=REGION, **kwargs)


def test_env_default_has_no_fixed_bucket(monkeypatch):
    monkeypatch.delenv("UAEF_S3_BUCKET", raising=False)
    assert StorageConfig.from_env().s3_bucket is None
    monkeypatch.setenv("UAEF_S3_BUCKET", "")
    assert StorageConfig.from_env().s3_bucket is None


def test_default_bucket_is_account_and_region_scoped(aws):
    store = DynamoS3Storage(config=_config())
    assert store._bucket == f"uaef-results-{MOTO_ACCOUNT}-{REGION}"


def test_explicit_bucket_is_kept(aws):
    store = DynamoS3Storage(config=_config(s3_bucket="my-bucket"))
    assert store._bucket == "my-bucket"


def test_every_s3_call_pins_bucket_owner(aws):
    store = DynamoS3Storage(config=_config(s3_bucket="my-bucket"))
    store._s3 = MagicMock()
    store._s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"{}")}

    store.create_bucket_if_not_exists()
    store._read_s3_results("exp")
    store._write_s3_results("exp", {})
    store._table = MagicMock()
    store.delete_experiment("exp")

    for call in ("head_bucket", "get_object", "put_object", "delete_object"):
        kwargs = getattr(store._s3, call).call_args.kwargs
        assert kwargs["ExpectedBucketOwner"] == MOTO_ACCOUNT, call


def test_foreign_bucket_fails_closed(aws):
    store = DynamoS3Storage(config=_config(s3_bucket="someone-elses-bucket"))
    store._s3 = MagicMock()
    store._s3.head_bucket.side_effect = ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadBucket"
    )
    with pytest.raises(PermissionError, match="owned by another account"):
        store.create_bucket_if_not_exists()
    store._s3.create_bucket.assert_not_called()


def test_missing_default_bucket_is_created(aws):
    store = DynamoS3Storage(config=_config())
    store.create_bucket_if_not_exists()
    names = [b["Name"] for b in boto3.client("s3", region_name=REGION).list_buckets()["Buckets"]]
    assert store._bucket in names


def test_get_experiment_uses_consistent_read(aws):
    # Read-after-write (evaluate(persist=True) then get_experiment) must not
    # race DynamoDB's default eventually consistent reads.
    store = DynamoS3Storage(config=_config(s3_bucket="my-bucket"))
    store._table = MagicMock()
    store._table.get_item.return_value = {"Item": {"experiment_id": "e1"}}
    assert store.get_experiment("e1") == {"experiment_id": "e1"}
    assert store._table.get_item.call_args.kwargs["ConsistentRead"] is True


def test_account_lookup_is_lazy(aws, monkeypatch):
    # DynamoDB-only use (list/get experiments) must not need STS
    calls = []
    real_client = boto3.client

    def counting_client(service, *args, **kwargs):
        calls.append(service)
        return real_client(service, *args, **kwargs)

    monkeypatch.setattr(boto3, "client", counting_client)
    store = DynamoS3Storage(config=_config())
    assert "sts" not in calls
    assert store.bucket == f"uaef-results-{MOTO_ACCOUNT}-{REGION}"
    assert calls.count("sts") == 1
    assert store.account_id == MOTO_ACCOUNT
    assert calls.count("sts") == 1  # cached


def test_full_results_read_from_stored_result_path(aws):
    # Experiments saved under an earlier bucket configuration stay readable
    store = DynamoS3Storage(config=_config(s3_bucket="new-bucket"))
    store._table = MagicMock()
    store._table.get_item.return_value = {
        "Item": {"experiment_id": "e1", "result_path": "s3://old-bucket/evaluations/e1.json"}
    }
    store._s3 = MagicMock()
    store._s3.get_object.return_value = {"Body": MagicMock(read=lambda: b'{"evaluations": [1]}')}

    assert store.get_full_results("e1") == {"evaluations": [1]}
    kwargs = store._s3.get_object.call_args.kwargs
    assert (kwargs["Bucket"], kwargs["Key"]) == ("old-bucket", "evaluations/e1.json")
    assert kwargs["ExpectedBucketOwner"] == MOTO_ACCOUNT


def test_forbidden_message_covers_missing_list_permission(aws):
    store = DynamoS3Storage(config=_config(s3_bucket="b"))
    store._s3 = MagicMock()
    store._s3.head_bucket.side_effect = ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadBucket"
    )
    with pytest.raises(PermissionError, match="s3:ListBucket"):
        store.create_bucket_if_not_exists()
