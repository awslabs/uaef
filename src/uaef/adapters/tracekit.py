# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracekit adapter for evaluating agents on Tracekit's signed run records.

Tracekit records each tool call an agent makes as a signed, hash-chained record, written by
a signer the agent does not control, and exports a run as a bundle (``.tkb``) that verifies
offline. This adapter verifies the bundle with Tracekit's own verifier before mapping it, so
an evaluation can state that it scored a record nobody edited after it was signed.

Requires the optional ``tracekit`` extra: ``pip install 'uaef[tracekit]'``.
"""

import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from uaef.adapters.base import AdapterTransformationError, BaseAdapter
from uaef.models.agent_trace import AgentTrace
from uaef.models.message import Message
from uaef.models.tool_call import ToolCall

# Human-readable label used in the actionable ImportError raised when the
# Tracekit verifier (shipped behind the 'tracekit' extra) is not installed.
_TRACEKIT_FEATURE = "Tracekit bundles"
_TRACEKIT_EXTRA = "tracekit"

# Fields copied from the signed records into a tool call's label, when the record has them.
_DECISION_FIELDS = (
    "tool_use_id",
    "tool",
    "decision",
    "rule_ids",
    "reasons",
    "policy_hash",
    "args_commitment",
)
_APPROVAL_FIELDS = ("decision", "approver", "approver_kind", "person", "self_approved")
_RESULT_FIELDS = ("ok", "output")


class TracekitAdapter(BaseAdapter):
    """
    Adapter for transforming a Tracekit v2 bundle (``.tkb``) to canonical AgentTrace format.

    The bundle is verified first, against a trust config that pins the signer's log key
    (``tracekit signer trust -o trust.json`` writes one). A bundle that does not verify
    raises ``AdapterTransformationError``, unless the adapter was made with
    ``allow_unverified=True``. The verdict is in ``metadata["tracekit"]["verification"]``
    either way.

    Expected raw_data::

        {
            "bundle_path": "run.tkb",     # tracekit export --v2 --run RUN -o run.tkb
            "trust_path": "trust.json",   # pins the key of the signer that signed it
        }

    What maps:

    - ``tool_calls``: one ``ToolCall`` per signed ``policy.decision``, in run order, with
      its tool name and the record's time. ``arguments`` stay empty and ``result`` None:
      the bundle holds commitments to them, not their values.
    - ``metadata["tracekit"]["calls"]``: one label per tool call, in the same order: the
      policy decision (``allow``, ``flag``, ``ask`` or ``deny``) and its rule ids, the
      answer a person or an agent approver gave to a held call, and the tool's result
      (``ok`` and the commitment to its output), as the signed records state them.
    - ``session_id``: the run's session (its own run id unless a parent run started it).

    Not mapped: messages (a bundle holds commitments to prompts and model output, never
    their text), tokens, cost and latency (not in the records), and a failed call's error
    text (committed to, not stored; its label has ``ok: False``).
    """

    def __init__(self, allow_unverified: bool = False):
        """
        Initialize the adapter.

        Args:
            allow_unverified: If True, map a bundle that fails verification instead of
                raising. Its verdict is still recorded in the trace's metadata.
        """
        self.allow_unverified = allow_unverified

    @property
    def name(self) -> str:
        """Return the adapter name."""
        return "tracekit"

    def transform_to_canonical(self, raw_data: Dict[str, Any]) -> AgentTrace:
        """
        Verify a Tracekit bundle and transform its run to canonical AgentTrace format.

        Args:
            raw_data: ``{"bundle_path": ..., "trust_path": ...}``

        Returns:
            Canonical AgentTrace object

        Raises:
            ImportError: If the 'tracekit' extra is not installed
            AdapterTransformationError: If the bundle does not verify (unless
                ``allow_unverified``), holds other than one run, or cannot be read
        """
        try:
            data = self._read_bundle(raw_data)
            first = data["records"][0]["event"]
            run_id = first["run_id"]
            try:
                trace_id = UUID(run_id)
            except ValueError:
                # Not a UUID: generate one; the run id is kept in metadata
                trace_id = uuid4()

            return AgentTrace(
                trace_id=trace_id,
                session_id=self._registered(data["records"]).get("session", run_id),
                messages=self.extract_messages(data),
                tool_calls=self.extract_tool_calls(data),
                metadata=self.extract_metadata(data),
                framework="tracekit",
                timestamp=self._parse_timestamp(first["ts"]),
            )

        except (AdapterTransformationError, ImportError):
            raise
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to transform Tracekit bundle: {str(e)}",
                adapter_name=self.name,
                original_error=e,
            )

    def extract_messages(self, raw_data: Dict[str, Any]) -> List[Message]:
        """
        Extract messages from a Tracekit bundle.

        A bundle holds commitments to prompts and model output, never their text, so there
        are no messages to extract.

        Args:
            raw_data: Tracekit bundle data

        Returns:
            An empty list
        """
        return []

    def extract_tool_calls(self, raw_data: Dict[str, Any]) -> List[ToolCall]:
        """
        Extract one tool call per signed policy decision, in run order.

        Args:
            raw_data: ``{"bundle_path": ..., "trust_path": ...}``, or the verified records
                as the adapter read them

        Returns:
            List of ToolCall objects in chronological order

        Raises:
            AdapterTransformationError: If tool call extraction fails
        """
        try:
            records = self._read_bundle(raw_data)["records"]
            return [
                ToolCall(
                    name=decision["event"]["data"]["tool"],
                    timestamp=self._parse_timestamp(decision["event"]["ts"]),
                )
                for decision, _, _ in self._calls(records)
            ]

        except (AdapterTransformationError, ImportError):
            raise
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract tool calls: {str(e)}",
                adapter_name=self.name,
                original_error=e,
            )

    def extract_metadata(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the verification verdict and the per-call labels.

        Returns ``{"tracekit": {...}}`` with:

        - ``verification``: ``verified``, ``integrity`` and ``assurance`` (the verifier's
          verdict), ``reason`` (its one-line result), ``summary`` (the short result
          ``tracekit verify`` prints) and ``verifier`` (the tracekit-ai version)
        - ``run_id``, ``tenant`` and ``agent`` (as the run registered it)
        - ``calls``: one label per tool call, in ``tool_calls`` order, citing its signed
          decision record by ``run_seq`` and ``hash``

        Args:
            raw_data: ``{"bundle_path": ..., "trust_path": ...}``, or the verified records
                as the adapter read them

        Returns:
            Dictionary of metadata

        Raises:
            AdapterTransformationError: If metadata extraction fails
        """
        try:
            data = self._read_bundle(raw_data)
            first = data["records"][0]["event"]

            calls = []
            for decision, result, approval in self._calls(data["records"]):
                fields = decision["event"]["data"]
                label = {k: fields[k] for k in _DECISION_FIELDS if k in fields}
                if approval is not None:
                    label["approval"] = {k: approval[k] for k in _APPROVAL_FIELDS if k in approval}
                if result is not None:
                    label["result"] = {k: result[k] for k in _RESULT_FIELDS if k in result}
                label["record"] = {
                    "run_seq": decision["event"]["run_seq"],
                    "hash": decision["hash"],
                }
                calls.append(label)

            return {
                "tracekit": {
                    "verification": data["verification"],
                    "run_id": first["run_id"],
                    "tenant": first["tenant"],
                    "agent": self._registered(data["records"]).get("agent"),
                    "calls": calls,
                }
            }

        except (AdapterTransformationError, ImportError):
            raise
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract metadata: {str(e)}",
                adapter_name=self.name,
                original_error=e,
            )

    def _read_bundle(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify the bundle with Tracekit's verifier and return its run's records.

        Data this adapter already read (with ``records``) is returned as it is, so
        ``transform_to_canonical`` verifies a bundle once.

        Args:
            raw_data: ``{"bundle_path": ..., "trust_path": ...}``

        Returns:
            ``{"records": [...], "verification": {...}}``

        Raises:
            ImportError: If the 'tracekit' extra is not installed
            AdapterTransformationError: If the bundle does not verify (unless
                ``allow_unverified``) or does not hold exactly one run
        """
        if "records" in raw_data:
            return raw_data
        self.validate_raw_data(raw_data, ["bundle_path", "trust_path"])

        from uaef.integrations._guards import require_integration

        tracekit = require_integration("tracekit", _TRACEKIT_FEATURE, extra=_TRACEKIT_EXTRA)
        from tracekit.verify import summary, v2  # type: ignore[import-untyped]

        runs: Dict[str, List[Dict[str, Any]]] = {}
        report, code = v2.verify(raw_data["bundle_path"], raw_data["trust_path"], runs=runs)
        text = io.StringIO()
        summary(report, code, report.integrity, report.assurance, text)
        lines = text.getvalue().splitlines()
        verification = {
            "verified": code == 0,
            "integrity": report.integrity,
            "assurance": report.assurance,
            "reason": lines[0],
            "summary": lines,
            "verifier": f"tracekit-ai {tracekit.__version__}",
        }

        if code != 0 and not self.allow_unverified:
            raise AdapterTransformationError(
                f"Bundle did not verify: {verification['reason']}",
                adapter_name=self.name,
            )
        if len(runs) != 1:
            raise AdapterTransformationError(
                f"Expected one run in the bundle, found {len(runs)}",
                adapter_name=self.name,
            )
        return {"records": next(iter(runs.values())), "verification": verification}

    @staticmethod
    def _calls(
        records: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
        """
        Pair each policy decision with its result and its approval, by the ids the records sign.

        A ``tool.result`` names the decision it completes (``decision_id``); an ``approval``
        names its ``approval.request``, which names the decision it holds.

        Args:
            records: The run's signed records, in run order

        Returns:
            ``(decision record, result data, approval data)`` per decision, in run order
        """
        events = [record["event"] for record in records]
        results = {
            e["data"]["decision_id"]: e["data"]
            for e in events
            if e["type"] == "tool.result" and "decision_id" in e["data"]
        }
        requests = {
            e["data"]["approval_id"]: e["data"]["decision_id"]
            for e in events
            if e["type"] == "approval.request"
        }
        approvals = {
            requests[e["data"]["approval_id"]]: e["data"]
            for e in events
            if e["type"] == "approval" and e["data"].get("approval_id") in requests
        }
        return [
            (
                record,
                results.get(record["event"]["data"].get("decision_id")),
                approvals.get(record["event"]["data"].get("decision_id")),
            )
            for record in records
            if record["event"]["type"] == "policy.decision"
        ]

    @staticmethod
    def _registered(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return the data of the run's ``run.registered`` record (empty if it has none)."""
        return next(
            (r["event"]["data"] for r in records if r["event"]["type"] == "run.registered"), {}
        )

    @staticmethod
    def _parse_timestamp(timestamp_str: str) -> datetime:
        """Parse a Tracekit record time (``2026-01-15T10:30:00.000000Z``, UTC)."""
        return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
