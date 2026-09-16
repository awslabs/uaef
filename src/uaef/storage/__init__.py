# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage layer for UAEF — DynamoDB + S3."""

from uaef.storage.dynamodb_s3 import DynamoS3Storage, get_storage, reset_storage

__all__ = ["DynamoS3Storage", "get_storage", "reset_storage"]
