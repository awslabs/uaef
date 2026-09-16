# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""StorageStack — Jobs DynamoDB table (+TTL) and payload S3 bucket.

This nested stack owns the UAEF Service's *own* transient state and references
(but does not own) the UAEF library's experiment persistence resources.

It provisions:

* The ``uaef-service-jobs`` DynamoDB table that holds async Job lifecycle state
  (Requirement 4.1). Partition key is ``jobId`` (String), billing is on-demand
  (pay-per-request), and DynamoDB TTL is enabled on the ``ttl`` attribute so
  expired Job records auto-delete (matching the ``ttl`` epoch-seconds field of
  the Job data model).
* The payload S3 bucket used for presigned PUT/GET of request and result
  payloads that exceed API Gateway's 10MB body limit (Requirement 7.1). The
  API Lambda presigns uploads into this bucket; lifecycle expiry keeps these
  transient objects from accumulating.

It also *references* the UAEF library's experiment table and results bucket
(``uaef-experiments`` / ``uaef-results``). The Worker Lambda calls ``uaef`` with
``persist=True``, so experiments and full result JSON are written through the
library's own persistence layer. This stack imports those resources by name so
later stacks (e.g. the Worker stack) can grant least-privilege access to them;
it deliberately does NOT (re)define their schema — they are owned by the library.

This module is part of the standalone ``uaef-service`` deployable app and does
NOT import the ``uaef`` package.

Requirements: 4.1, 7.1
"""

from __future__ import annotations

from typing import Optional

from aws_cdk import (
    NestedStack,
    RemovalPolicy,
    Duration,
    aws_dynamodb as dynamodb,
    aws_s3 as s3,
)
from constructs import Construct

from .constants import (
    JOBS_TABLE_NAME,
    PAYLOAD_BUCKET_NAME,
    UAEF_EXPERIMENT_TABLE_NAME,
    UAEF_RESULTS_BUCKET_NAME,
)
from .cors import resolve_cors_origins


class StorageStack(NestedStack):
    """Jobs table (+TTL) and payload bucket; references UAEF persistence.

    Attributes:
        jobs_table: The ``uaef-service-jobs`` DynamoDB table holding async Job
            lifecycle state (PK ``jobId``, TTL on ``ttl``). Written by the API
            Lambda (``PENDING``) and the Worker Lambda (state transitions).
        payload_bucket: The S3 bucket for presigned upload/download of large
            request and result payloads offloaded from API Gateway's 10MB limit.
        uaef_experiment_table: A reference to the UAEF library's experiment
            metadata table (owned by the library, imported here for grants).
        uaef_results_bucket: A reference to the UAEF library's full-result S3
            bucket (owned by the library, imported here for grants).
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        cors_origin: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- CORS allowed origins (security review M-01) -------------------
        # The browser UI uploads (presigned PUT) and downloads (presigned GET)
        # directly to/from the payload bucket, so it needs a CORS allow-list.
        #
        # Preferred path (with_ui): the parent passes ``cors_origin`` — the
        # deployed UI's own CloudFront origin, as a CloudFormation token
        # resolved at deploy time. No wildcard is ever needed: the real origin
        # is known within the single deploy. Extra origins (e.g. a localhost dev
        # server) are still read from the ``api_cors_allowed_origins`` context
        # and merged in, so with_ui can allow a dev server alongside the
        # deployed UI. Kept in lock-step with ApiStack, which merges the same
        # two sources, so the bucket and API allow-lists never drift.
        #
        # Fallback path (no derived UI origin — e.g. with_eks, or a UI-less
        # API-only deploy): the origin cannot be derived, so it MUST be supplied
        # explicitly via the ``api_cors_allowed_origins`` context. A wildcard is
        # still rejected there unless explicitly acknowledged.
        payload_bucket_cors_origins = resolve_cors_origins(self, cors_origin)

        # --- Jobs table (owned by this service) ---------------------------
        # Async Job lifecycle state. Partition key ``jobId`` (String); no sort
        # key. On-demand billing absorbs the spiky, low-volume job-control
        # traffic without capacity planning. DynamoDB TTL on the ``ttl``
        # attribute (epoch seconds) auto-expires finished Job records.
        self.jobs_table = dynamodb.Table(
            self,
            "JobsTable",
            table_name=JOBS_TABLE_NAME,
            partition_key=dynamodb.Attribute(
                name="jobId",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            point_in_time_recovery=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # --- Payload bucket (owned by this service) -----------------------
        # Holds transient request/result payloads that exceed API Gateway's
        # 10MB body limit. The API Lambda issues short-lived presigned PUT/GET
        # URLs scoped to single keys. Objects are transient, so a lifecycle
        # rule expires them; the bucket is private and encrypted.
        self.payload_bucket = s3.Bucket(
            self,
            "PayloadBucket",
            bucket_name=PAYLOAD_BUCKET_NAME,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="expire-transient-payloads",
                    enabled=True,
                    expiration=Duration.days(7),
                    abort_incomplete_multipart_upload_after=Duration.days(1),
                ),
            ],
            # The deployed browser UI uploads data files (presigned PUT) and
            # downloads results (presigned GET) directly to/from this bucket,
            # so it needs CORS. Security review M-01: origins are the same
            # deployer-supplied allow-list ApiStack requires (resolved above),
            # not a wildcard.
            cors=[
                s3.CorsRule(
                    allowed_methods=[
                        s3.HttpMethods.GET,
                        s3.HttpMethods.PUT,
                        s3.HttpMethods.HEAD,
                    ],
                    allowed_origins=payload_bucket_cors_origins,
                    allowed_headers=["*"],
                    exposed_headers=["ETag"],
                    max_age=3000,
                ),
            ],
            removal_policy=RemovalPolicy.RETAIN,
        )

        # --- UAEF library persistence (owned by this stack) ----------------
        # The experiment table and results bucket are used by the library's
        # persistence layer (persist=True) and the Reducer (combined experiment
        # persistence). Previously these were only imported by name (assuming the
        # library would auto-create them), but that requires CreateTable
        # permission on every Lambda. Creating them here via CDK is cleaner and
        # ensures they exist before any evaluation runs.
        self.uaef_experiment_table = dynamodb.Table(
            self,
            "UaefExperimentTable",
            table_name=UAEF_EXPERIMENT_TABLE_NAME,
            partition_key=dynamodb.Attribute(
                name="experiment_id",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        # Security review H-02: GSI on `created_by` so the service can list a
        # caller's own experiments via a scoped query (DynamoS3Storage's
        # CREATED_BY_INDEX_NAME) instead of a full unscoped table scan. Rows
        # written before this fix have no `created_by` attribute and are
        # simply absent from this index — never returned by the owner-scoped
        # query, and (per the API-side owner check on the direct-read routes)
        # not otherwise readable either. That is the intended fail-closed
        # behavior for pre-existing, ownerless rows: no owner recorded, no
        # access, rather than treating them as ambiguously shared.
        self.uaef_experiment_table.add_global_secondary_index(
            index_name="created_by-index",
            partition_key=dynamodb.Attribute(
                name="created_by",
                type=dynamodb.AttributeType.STRING,
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )
        self.uaef_results_bucket = s3.Bucket(
            self,
            "UaefResultsBucket",
            bucket_name=UAEF_RESULTS_BUCKET_NAME,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,
            removal_policy=RemovalPolicy.RETAIN,
        )

    @property
    def jobs_table_name(self) -> str:
        """The Jobs table name (consumed by the API and Worker Lambdas)."""
        return self.jobs_table.table_name

    @property
    def payload_bucket_name(self) -> str:
        """The payload bucket name (presigned by the API Lambda)."""
        return self.payload_bucket.bucket_name
