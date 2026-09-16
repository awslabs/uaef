# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DynamoDB + S3 storage layer for UAEF experiments and evaluation results.

DynamoDB: One row per experiment with average scores and metadata.
S3: Full evaluation results as JSON per experiment.
"""

import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

import boto3
from botocore.exceptions import ClientError

from uaef.config import StorageConfig, get_config

logger = logging.getLogger(__name__)


def _decimal_default(obj: Any) -> Any:
    """JSON serializer for Decimal types from DynamoDB."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _to_dynamo(value: Any) -> Any:
    """Convert Python floats to Decimal for DynamoDB compatibility."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _to_dynamo(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_dynamo(v) for v in value]
    return value


class DynamoS3Storage:
    """
    Storage layer using DynamoDB for experiment metadata and S3 for full results.

    DynamoDB schema (one row per experiment):
        - experiment_id (PK): str
        - experiment_name: str
        - experiment_objective: str | None
        - created_at: str (ISO 8601)
        - updated_at: str (ISO 8601)
        - evaluation_count: int
        - average_scores: dict  (metric_name -> average score)
        - overall_average_score: Decimal
        - result_path: str (s3://bucket/prefix/experiment_id.json)
        - metadata: dict

    S3 object (one JSON file per experiment):
        {
            "experiment_id": "...",
            "experiment_name": "...",
            "evaluations": [ ... full EvaluationResult dicts ... ]
        }
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """Initialize DynamoDB + S3 storage.

        Args:
            config: Storage configuration. Uses global config if None.
        """
        self.config = config or get_config().storage
        region = self.config.region or get_config().aws.region

        self._dynamodb = boto3.resource("dynamodb", region_name=region)
        self._s3 = boto3.client("s3", region_name=region)
        self._table_name = self.config.dynamodb_table_name
        self._bucket = self.config.s3_bucket
        self._prefix = self.config.s3_prefix
        self._table = self._dynamodb.Table(self._table_name)

        logger.info(
            f"DynamoS3Storage initialized: table={self._table_name}, "
            f"bucket={self._bucket}, prefix={self._prefix}"
        )

    # ------------------------------------------------------------------
    # Table provisioning
    # ------------------------------------------------------------------

    def create_table_if_not_exists(self) -> None:
        """Create the DynamoDB table if it does not already exist."""
        try:
            self._dynamodb.create_table(
                TableName=self._table_name,
                KeySchema=[
                    {"AttributeName": "experiment_id", "KeyType": "HASH"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "experiment_id", "AttributeType": "S"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            self._table.wait_until_exists()
            logger.info(f"Created DynamoDB table '{self._table_name}'")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceInUseException":
                logger.debug(f"Table '{self._table_name}' already exists")
            else:
                raise

    def create_bucket_if_not_exists(self) -> None:
        """Create the S3 bucket if it does not already exist."""
        try:
            self._s3.head_bucket(Bucket=self._bucket)
            logger.debug(f"S3 bucket '{self._bucket}' already exists")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("404", "NoSuchBucket"):
                region = self.config.region or get_config().aws.region
                create_kwargs = {"Bucket": self._bucket}
                # us-east-1 doesn't need LocationConstraint
                if region != "us-east-1":
                    create_kwargs["CreateBucketConfiguration"] = {
                        "LocationConstraint": region
                    }
                self._s3.create_bucket(**create_kwargs)
                logger.info(f"Created S3 bucket '{self._bucket}'")
            else:
                raise

    # ------------------------------------------------------------------
    # S3 helpers
    # ------------------------------------------------------------------

    def _s3_key(self, experiment_id: str) -> str:
        return f"{self._prefix}{experiment_id}.json"

    def _result_path(self, experiment_id: str) -> str:
        return f"s3://{self._bucket}/{self._s3_key(experiment_id)}"

    def _read_s3_results(self, experiment_id: str) -> Dict[str, Any]:
        """Read the full results JSON from S3."""
        try:
            resp = self._s3.get_object(
                Bucket=self._bucket,
                Key=self._s3_key(experiment_id),
            )
            return json.loads(resp["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return {
                    "experiment_id": experiment_id,
                    "evaluations": [],
                }
            raise

    def _write_s3_results(self, experiment_id: str, data: Dict[str, Any]) -> None:
        """Write the full results JSON to S3."""
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._s3_key(experiment_id),
            Body=json.dumps(data, default=_decimal_default, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def save_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        evaluation_results: List[Dict[str, Any]],
        experiment_objective: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save or update an experiment with its evaluation results.

        Computes average scores from all evaluation results, writes the full
        results to S3, and upserts the DynamoDB row.

        Args:
            experiment_id: Unique experiment identifier.
            experiment_name: Human-readable name.
            evaluation_results: List of EvaluationResult dicts (serialized).
            experiment_objective: Optional description/objective.
            metadata: Optional extra metadata.
            created_by: Security review H-02: the identity (e.g. Cognito
                ``sub``) of the caller who created this experiment, used by
                the UAEF Service to scope ``list_experiments``/read access to
                the owner. Optional — library-only callers with no per-caller
                identity concept (no service in front of them) may omit it,
                in which case the experiment has no owner attribute and is
                only reachable directly by ``experiment_id``, never via an
                owner-scoped listing. Preserved across updates (an existing
                experiment's ``created_by`` is never overwritten by a later
                save with a different or missing value), matching how
                ``created_at`` is already preserved below.

        Returns:
            The DynamoDB item dict that was written.
        """
        now = datetime.utcnow().isoformat()

        # --- Compute average scores ---
        metric_totals: Dict[str, List[float]] = {}
        overall_scores: List[float] = []

        for er in evaluation_results:
            if "overall_score" in er and er["overall_score"] is not None:
                overall_scores.append(float(er["overall_score"]))
            for dim in er.get("dimension_results", []):
                for ms in dim.get("metric_scores", []):
                    name = ms.get("metric_name", "")
                    score = ms.get("score")
                    if name and score is not None:
                        metric_totals.setdefault(name, []).append(float(score))

        average_scores = {
            name: round(sum(vals) / len(vals), 4)
            for name, vals in metric_totals.items()
        }
        overall_avg = (
            round(sum(overall_scores) / len(overall_scores), 4)
            if overall_scores
            else 0.0
        )

        # --- Write full results to S3 ---
        s3_data = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "experiment_objective": experiment_objective,
            "evaluations": evaluation_results,
        }
        self._write_s3_results(experiment_id, s3_data)

        # --- Upsert DynamoDB row ---
        item = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "experiment_objective": experiment_objective or "",
            "created_at": now,
            "updated_at": now,
            "evaluation_count": len(evaluation_results),
            "average_scores": _to_dynamo(average_scores),
            "overall_average_score": _to_dynamo(overall_avg),
            "result_path": self._result_path(experiment_id),
            "metadata": _to_dynamo(metadata or {}),
        }
        if created_by:
            item["created_by"] = created_by

        # Preserve original created_at (and created_by — security review
        # H-02: an experiment's owner must not change on a later save that
        # omits it or names someone else) if the experiment already exists.
        existing = self.get_experiment(experiment_id)
        if existing and "created_at" in existing:
            item["created_at"] = existing["created_at"]
        if existing and existing.get("created_by"):
            item["created_by"] = existing["created_by"]

        self._table.put_item(Item=item)
        logger.info(
            f"Saved experiment {experiment_id}: "
            f"{len(evaluation_results)} evaluations, "
            f"overall_avg={overall_avg}"
        )
        return item

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment metadata from DynamoDB.

        Args:
            experiment_id: Experiment ID.

        Returns:
            DynamoDB item dict or None if not found.
        """
        try:
            resp = self._table.get_item(Key={"experiment_id": experiment_id})
            return resp.get("Item")
        except ClientError:
            return None

    def get_experiment_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find an experiment by name via DynamoDB scan.

        Args:
            name: Experiment name to search for.

        Returns:
            DynamoDB item dict or None if not found.
        """
        resp = self._table.scan(
            FilterExpression="experiment_name = :n",
            ExpressionAttributeValues={":n": name},
            Limit=1,
        )
        items = resp.get("Items", [])
        return items[0] if items else None

    def get_full_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get the full evaluation results from S3.

        Args:
            experiment_id: Experiment ID.

        Returns:
            Full results dict including all evaluations.
        """
        return self._read_s3_results(experiment_id)

    #: Name of the GSI on ``created_by`` used to scope ``list_experiments`` to
    #: a single owner (security review H-02). Must match the index name the
    #: CDK-provisioned table declares (see ``uaef-service/infra/storage_stack.py``).
    CREATED_BY_INDEX_NAME = "created_by-index"

    def list_experiments(self, created_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List experiments from DynamoDB.

        Args:
            created_by: Security review H-02: when provided, scopes the
                listing to experiments owned by this identity (e.g. a
                Cognito ``sub``), via the ``created_by`` GSI — never a
                client-side filter over a full scan, so an experiment
                belonging to another caller is never even read out of the
                table. When ``None`` (the default — unchanged library-only
                behavior), lists every experiment via a full table scan, as
                before; library callers with no per-caller identity concept
                have no owner boundary to enforce. The UAEF Service (which
                does have caller identity) always passes ``created_by``.

        Returns:
            List of experiment item dicts.
        """
        items: List[Dict[str, Any]] = []
        if created_by:
            kwargs: Dict[str, Any] = {
                "IndexName": self.CREATED_BY_INDEX_NAME,
                "KeyConditionExpression": "created_by = :cb",
                "ExpressionAttributeValues": {":cb": created_by},
            }
            resp = self._table.query(**kwargs)
            items.extend(resp.get("Items", []))
            while "LastEvaluatedKey" in resp:
                resp = self._table.query(
                    **kwargs, ExclusiveStartKey=resp["LastEvaluatedKey"]
                )
                items.extend(resp.get("Items", []))
            return items

        resp = self._table.scan()
        items.extend(resp.get("Items", []))
        while "LastEvaluatedKey" in resp:
            resp = self._table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
            items.extend(resp.get("Items", []))
        return items

    def delete_experiment(self, experiment_id: str) -> None:
        """
        Delete an experiment from DynamoDB and S3.

        Args:
            experiment_id: Experiment ID to delete.
        """
        self._table.delete_item(Key={"experiment_id": experiment_id})
        try:
            self._s3.delete_object(
                Bucket=self._bucket,
                Key=self._s3_key(experiment_id),
            )
        except ClientError:
            pass
        logger.info(f"Deleted experiment {experiment_id}")


# ------------------------------------------------------------------
# Global instance
# ------------------------------------------------------------------

_storage: Optional[DynamoS3Storage] = None


def get_storage(config: Optional[StorageConfig] = None) -> DynamoS3Storage:
    """Get or create the global DynamoS3Storage instance."""
    global _storage
    if _storage is None:
        _storage = DynamoS3Storage(config=config)
    return _storage


def reset_storage() -> None:
    """Reset the global storage instance."""
    global _storage
    _storage = None
