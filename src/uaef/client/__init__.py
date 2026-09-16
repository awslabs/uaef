# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Optional UAEF client SDK.

``UAEFClient`` is a thin HTTP wrapper that gives remote users the same
ergonomics as the local ``uaef.api.evaluate`` / ``batch_evaluate`` functions
while hiding the asynchronous POST-then-poll mechanics of the UAEF service.

This module ships with the library behind the ``client`` extra::

    pip install 'uaef[client]'

The only third-party dependency is ``requests``. It is imported lazily/guarded
so that importing the core ``uaef`` package never fails when the ``client``
extra is not installed.

Example:
    >>> from uaef.client import UAEFClient
    >>> client = UAEFClient(endpoint="https://api.example.com", token=jwt)
    >>> result = client.evaluate(trace=trace, ground_truth=gt, persist=True,
    ...                          experiment_name="remote-run")
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union


# Terminal job states (mirror of the service-side async job state machine).
_TERMINAL_STATES = frozenset({"COMPLETED", "FAILED"})

# Defaults mandated by Requirements 10.5 (fixed 2s poll interval) and
# 10.6 (15-minute / 900s timeout).
_DEFAULT_POLL_INTERVAL = 2.0
_DEFAULT_TIMEOUT = 900.0

# Per-request HTTP timeouts as (connect, read) seconds. These are DISTINCT from
# the overall poll ``timeout`` above: they bound each individual HTTP call so a
# hung/unresponsive connection can't block forever (bandit B113). They do NOT
# cap total job time — that remains governed by the poll ``timeout``.
_DEFAULT_REQUEST_TIMEOUT = (10.0, 60.0)     # quick API calls: submit / presign / status poll
_DEFAULT_TRANSFER_TIMEOUT = (10.0, 300.0)   # data transfer: S3 payload upload / result download

# API Gateway caps request bodies at 10MB. A body at or below this limit is
# sent inline; anything larger must be offloaded to S3 via the presigned
# upload path and referenced by key (Requirements 7.x, 10.7).
_API_GATEWAY_BODY_LIMIT = 10 * 1024 * 1024  # 10 MiB

# For each async submit path, the inline payload field that is offloaded to S3
# when the request is oversized, paired with the reference field that carries
# the resulting object key back to the service.
_OFFLOAD_FIELDS: Dict[str, Tuple[str, str]] = {
    "/evaluate": ("trace", "traceRef"),
    "/batch-evaluate": ("traces", "tracesRef"),
}


def _json_size(payload: Any) -> int:
    """Return the size, in bytes, of ``payload`` serialized as compact JSON.

    Uses the same UTF-8 JSON encoding that is sent over the wire so the
    measurement reflects the actual API Gateway request-body size.
    """
    return len(json.dumps(payload).encode("utf-8"))


def _drop_none(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``body`` without keys whose value is ``None``.

    Keeps the submitted request payload aligned with the service-side
    ``EvaluateRequest`` / ``BatchEvaluateRequest`` schemas, which default
    optional fields to ``None``: omitting them lets the server apply its own
    defaults instead of receiving explicit nulls.
    """
    return {key: value for key, value in body.items() if value is not None}


def _require_requests():
    """Import ``requests`` lazily, raising an actionable error if missing.

    Keeping the import out of module top-level ensures that importing the core
    ``uaef`` package (which does not depend on ``requests``) never breaks when
    the optional ``client`` extra is not installed (Requirement 10.1).
    """
    try:
        import requests  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via guard test
        raise ImportError(
            "The UAEFClient SDK requires the 'client' extra. "
            "Install with: pip install 'uaef[client]'"
        ) from exc
    return requests


class JobFailedError(RuntimeError):
    """Raised when a polled job reaches a ``FAILED`` terminal state.

    Carries the job's error ``code`` and ``message`` so callers can inspect the
    failure programmatically (Requirement 10.8).
    """

    def __init__(self, code: Optional[str], message: Optional[str], job_id: str):
        self.code = code
        self.message = message
        self.job_id = job_id
        super().__init__(
            f"Job {job_id} failed [{code or 'UNKNOWN'}]: {message or 'no message provided'}"
        )


class UAEFClient:
    """Synchronous client for the UAEF HTTP service.

    The client submits an evaluation request, polls the job-status endpoint at a
    fixed interval until the job reaches a terminal state, and returns the
    result, hiding the POST-and-poll mechanics from the caller.

    Args:
        endpoint: Base URL of the UAEF service (e.g. ``https://api.example.com``).
        token: Cognito-issued JWT used as a bearer token on every request.
        poll_interval: Seconds between job-status polls. Fixed at 2.0s per
            Requirement 10.5.
        timeout: Maximum seconds to wait for a terminal state before raising a
            timeout error. Defaults to 900s (15 minutes) per Requirement 10.6.
            This is the OVERALL job wait, spread across many short polls — it is
            not the per-request timeout below.
        request_timeout: Per-request (connect, read) timeout in seconds for the
            quick API calls (submit, presign, status poll). A single float sets
            both. Prevents an unresponsive connection from hanging forever
            without capping total job time. Defaults to (10, 60).
        transfer_timeout: Per-request (connect, read) timeout in seconds for the
            data-transfer calls (S3 payload upload and result download), which
            may move large payloads. Defaults to (10, 300).
    """

    def __init__(
        self,
        endpoint: str,
        token: str,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        request_timeout: Union[float, Tuple[float, float]] = _DEFAULT_REQUEST_TIMEOUT,
        transfer_timeout: Union[float, Tuple[float, float]] = _DEFAULT_TRANSFER_TIMEOUT,
    ) -> None:
        # Validate the optional dependency up front so construction fails fast
        # with an actionable message rather than at first network call.
        _require_requests()

        if not endpoint:
            raise ValueError("endpoint is required")
        if not token:
            raise ValueError("token is required")

        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.request_timeout = request_timeout
        self.transfer_timeout = transfer_timeout

    # ------------------------------------------------------------------
    # Public API (evaluate / batch_evaluate implemented in task 5.2)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        trace,
        ground_truth=None,
        adapter: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        context: Optional[List[str]] = None,
        persist: bool = False,
        experiment_name: Optional[str] = None,
        experiment_objective: Optional[str] = None,
    ):
        """Submit a single evaluation and return the result.

        Mirrors the core arguments of ``uaef.api.evaluate``. The call is made
        against the service's ``POST /evaluate`` endpoint, which is
        asynchronous: the request body is submitted, a ``jobId`` is returned,
        and this method polls the job until it reaches a terminal state before
        returning the completed result payload (Requirements 10.2, 10.4).

        Args:
            trace: Agent trace to evaluate (canonical or raw output dict).
            ground_truth: Expected correct outputs (optional).
            adapter: Framework adapter name ("langgraph", "bedrock", etc.).
            metrics: Flat list of curated, server-side metric names.
            context: List of context documents (optional).
            persist: If True, the service creates and persists an experiment.
            experiment_name: Name for the new experiment when ``persist`` is set.
            experiment_objective: Description/objective for the experiment.

        Returns:
            The completed evaluation result payload (the full result fetched
            from ``resultRef`` when present, otherwise the inline
            ``resultSummary``).
        """
        body = _drop_none(
            {
                "trace": trace,
                "ground_truth": ground_truth,
                "adapter": adapter,
                "metrics": metrics,
                "context": context,
                "persist": persist,
                "experiment_name": experiment_name,
                "experiment_objective": experiment_objective,
            }
        )

        job_id = self._submit("/evaluate", body)
        status_record = self._poll(job_id)
        return self._extract_result(status_record)

    def batch_evaluate(
        self,
        traces,
        ground_truths=None,
        adapter: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        max_workers: int = 4,
        persist: bool = False,
        experiment_name: Optional[str] = None,
        experiment_objective: Optional[str] = None,
    ):
        """Submit a batch evaluation and return the result.

        Mirrors the core arguments of ``uaef.api.batch_evaluate``. The call is
        made against the service's ``POST /batch-evaluate`` endpoint, which is
        asynchronous: the request body is submitted, a ``jobId`` is returned,
        and this method polls the job until it reaches a terminal state before
        returning the completed result payload (Requirements 10.3, 10.4).

        Args:
            traces: List of agent traces to evaluate.
            ground_truths: List of expected correct outputs (optional).
            adapter: Framework adapter name.
            metrics: Flat list of curated, server-side metric names.
            max_workers: Maximum number of parallel workers on the service side.
            persist: If True, the service creates and persists an experiment.
            experiment_name: Name for the new experiment when ``persist`` is set.
            experiment_objective: Description/objective for the experiment.

        Returns:
            The completed batch evaluation result payload (the full result
            fetched from ``resultRef`` when present, otherwise the inline
            ``resultSummary``).
        """
        body = _drop_none(
            {
                "traces": traces,
                "ground_truths": ground_truths,
                "adapter": adapter,
                "metrics": metrics,
                "max_workers": max_workers,
                "persist": persist,
                "experiment_name": experiment_name,
                "experiment_objective": experiment_objective,
            }
        )

        job_id = self._submit("/batch-evaluate", body)
        status_record = self._poll(job_id)
        return self._extract_result(status_record)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Fetch the current status record for a job."""
        return self._get_job_status(job_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _submit(self, path: str, body: Dict[str, Any]) -> str:
        """POST ``body`` to ``path``, auto-offloading oversized payloads.

        API Gateway rejects request bodies larger than 10MB. When the
        serialized body exceeds that limit, the large inline payload (the
        ``trace`` for ``/evaluate``, the ``traces`` for ``/batch-evaluate``) is
        uploaded to S3 via the presigned upload path and replaced with its
        object key reference (``traceRef`` / ``tracesRef``) before submitting,
        keeping the request within the body limit (Requirement 10.7).

        Args:
            path: Service path, e.g. ``/evaluate`` or ``/batch-evaluate``.
            body: JSON-serializable request body.

        Returns:
            The ``jobId`` of the created job.

        Raises:
            RuntimeError: If the response does not contain a ``jobId``.
        """
        requests = _require_requests()

        body = self._offload_if_oversized(path, body)

        url = f"{self.endpoint}/{path.lstrip('/')}"
        response = requests.post(
            url, json=body, headers=self._headers(), timeout=self.request_timeout
        )
        response.raise_for_status()

        data = response.json()
        job_id = data.get("jobId")
        if not job_id:
            raise RuntimeError(
                f"Submit to {path} did not return a jobId (got: {data!r})"
            )
        return job_id

    def _offload_if_oversized(
        self, path: str, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Offload the large inline payload to S3 when ``body`` exceeds 10MB.

        When the JSON-serialized ``body`` is within the API Gateway 10MB body
        limit, it is returned unchanged. Otherwise the offloadable inline field
        for ``path`` (``trace`` for ``/evaluate``, ``traces`` for
        ``/batch-evaluate``) is uploaded to S3 via a freshly issued presigned
        PUT URL and replaced by its object key in the matching reference field
        (``traceRef`` / ``tracesRef``). The request then carries only the small
        reference inline (Requirement 10.7).

        If the body is oversized but the path has no offloadable field, or that
        field is absent/empty, the body is returned unchanged and submitted
        as-is (the service responds with a 413 in that case).

        Args:
            path: Service path being submitted to.
            body: The request body to inspect and possibly rewrite.

        Returns:
            The original body when within the limit, otherwise a new body dict
            with the large payload replaced by an S3 key reference.
        """
        if _json_size(body) <= _API_GATEWAY_BODY_LIMIT:
            return body

        # Normalize to a leading-slash, no-trailing-slash path for lookup.
        normalized_path = "/" + path.strip("/")
        offload = _OFFLOAD_FIELDS.get(normalized_path)
        if offload is None:
            # No known large field to offload for this path; let the service
            # decide (it will return a 413 for an oversized inline body).
            return body

        inline_field, ref_field = offload
        payload = body.get(inline_field)
        if payload is None:
            # Nothing to offload (e.g. caller already passed a *Ref); submit
            # as-is and let the service respond.
            return body

        key = self._upload_payload(payload)

        # Rebuild the body without the bulky inline field, referencing the
        # uploaded object by key instead.
        new_body = {k: v for k, v in body.items() if k != inline_field}
        new_body[ref_field] = key
        return new_body

    def _upload_payload(self, payload: Any) -> str:
        """Upload ``payload`` to S3 via a presigned PUT URL and return its key.

        Requests a presigned PUT URL and unique object key from the service's
        ``POST /payloads`` endpoint, then PUTs the JSON-serialized payload
        directly to S3 using that URL. The presigned URL carries its own auth,
        so the bearer token is intentionally not attached to the upload
        (Requirement 10.7).

        Args:
            payload: The portion of the request to store in S3 (e.g. the trace
                or list of traces).

        Returns:
            The S3 object key to reference in the subsequent submit request.

        Raises:
            RuntimeError: If the presign response omits an upload URL or key.
        """
        requests = _require_requests()

        presign_url = f"{self.endpoint}/payloads"
        presign_response = requests.post(
            presign_url, headers=self._headers(), timeout=self.request_timeout
        )
        presign_response.raise_for_status()

        presign = presign_response.json()
        upload_url = presign.get("uploadUrl")
        key = presign.get("key")
        if not upload_url or not key:
            raise RuntimeError(
                f"Presign request to /payloads did not return an uploadUrl and "
                f"key (got: {presign!r})"
            )

        # PUT the serialized payload straight to S3. The presigned URL is
        # already authenticated, so no Authorization header is sent.
        put_response = requests.put(
            upload_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=self.transfer_timeout,
        )
        put_response.raise_for_status()

        return key

    def _extract_result(self, status_record: Dict[str, Any]) -> Any:
        """Return the result payload from a terminal job-status record.

        On ``COMPLETED`` the service populates ``resultRef`` (a presigned GET
        URL for the full result JSON stored in S3) and/or an inline
        ``resultSummary``. This prefers the full result behind ``resultRef``,
        fetching and parsing it, and falls back to the inline ``resultSummary``
        when no reference is present (Requirement 10.4).

        Args:
            status_record: The terminal job-status record returned by ``_poll``.

        Returns:
            The parsed full result payload when ``resultRef`` is present,
            otherwise the inline ``resultSummary``, otherwise the full status
            record as a last resort.
        """
        result_ref = status_record.get("resultRef")
        if result_ref:
            requests = _require_requests()
            # resultRef is a presigned GET URL on COMPLETED, so it carries its
            # own auth — do not attach the bearer token.
            response = requests.get(result_ref, timeout=self.transfer_timeout)
            response.raise_for_status()
            return response.json()

        result_summary = status_record.get("resultSummary")
        if result_summary is not None:
            return result_summary

        return status_record

    def _get_job_status(self, job_id: str) -> Dict[str, Any]:
        """GET ``/jobs/{job_id}`` and return the parsed status record."""
        requests = _require_requests()

        url = f"{self.endpoint}/jobs/{job_id}"
        response = requests.get(
            url, headers=self._headers(), timeout=self.request_timeout
        )
        response.raise_for_status()
        return response.json()

    def _poll(self, job_id: str) -> Dict[str, Any]:
        """Poll the job-status endpoint until the job reaches a terminal state.

        Polls GET ``/jobs/{job_id}`` every ``poll_interval`` seconds (fixed 2s,
        Requirement 10.5) until the job is ``COMPLETED`` or ``FAILED``.

        Returns:
            The terminal job-status record when the job reaches ``COMPLETED``.

        Raises:
            TimeoutError: If the job does not reach a terminal state within
                ``timeout`` seconds (Requirement 10.6).
            JobFailedError: If the job reaches ``FAILED``, carrying the job's
                error code and message (Requirement 10.8).
        """
        deadline = time.monotonic() + self.timeout

        while True:
            status_record = self._get_job_status(job_id)
            status = status_record.get("status")

            if status in _TERMINAL_STATES:
                if status == "FAILED":
                    error = status_record.get("error") or {}
                    raise JobFailedError(
                        code=error.get("code"),
                        message=error.get("message"),
                        job_id=job_id,
                    )
                return status_record

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Job {job_id} did not reach a terminal state within "
                    f"{self.timeout:.0f}s"
                )

            time.sleep(self.poll_interval)


__all__ = ["UAEFClient", "JobFailedError"]
