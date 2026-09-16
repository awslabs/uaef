# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security review M-02: opt-in PII redaction of the persisted/UI copy.

``_build_ui_result`` shapes the payload that is persisted and rendered in the
UI. When ``UAEF_REDACT_PII_ON_PERSIST`` is set, every user-visible string in
that payload must be redacted.

These tests exist because the multi-turn work added a per-turn breakdown
(``row["turns"][n]["query"/"response"/"expected"]``) *after* M-02 was written.
Those turn fields are persisted and rendered exactly like the conversation-level
ones, so redacting only the top-level fields would leave PII in the payload —
a silent hole in the control. The multi-turn case is therefore asserted
field-by-field rather than just spot-checked.

Two properties are pinned:

1. Off by default — an existing deployment sees byte-identical output, so
   enabling redaction is the only thing that changes behavior.
2. When on, NO user-visible field in the payload carries the PII, at either
   the conversation level or the per-turn level.
"""

from __future__ import annotations

import importlib
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from uaef.models.agent_trace import AgentTrace
from uaef.models.ground_truth import GroundTruth
from uaef.models.message import Message, MessageRole

# An address redact_for_storage's default detector recognises (EMAIL).
EMAIL = "alice.smith@example.com"


def _load_worker(monkeypatch, *, redact: bool):
    """Import handlers.worker with the M-02 env var in the desired state.

    ``_pii_redaction_enabled()`` reads the environment at call time, but the
    module is reloaded anyway so the test cannot be order-dependent on a
    previously imported copy.
    """
    monkeypatch.setenv("UAEF_REDACT_PII_ON_PERSIST", "true" if redact else "false")
    sys.modules.pop("handlers.worker", None)
    return importlib.import_module("handlers.worker")


def _assistant_trace(text: str) -> AgentTrace:
    return AgentTrace(
        trace_id=uuid4(),
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content=text,
                timestamp=datetime.now(timezone.utc),
            )
        ],
        tool_calls=[],
    )


def _stub_result() -> MagicMock:
    """A result object with no dimension data — scores are irrelevant here."""
    result = MagicMock()
    result.dimension_results = []
    result.overall_score = 0.5
    result.passed = True
    result.experiment_id = None
    return result


def _multi_turn_ui_result(worker) -> Dict[str, Any]:
    """One 2-turn conversation where every text field contains the email."""
    turn_1 = _assistant_trace(f"Turn one response {EMAIL}")
    turn_2 = _assistant_trace(f"Turn two response {EMAIL}")
    session = {
        "session_id": "session_001",
        "per_turn_traces": [turn_1, turn_2],
        "full_trace": turn_2,
    }
    ground_truth = GroundTruth(
        expected_output=f"Expected answer {EMAIL}",
        expected_arguments={
            "per_turn_expected": [f"Expected 1 {EMAIL}", f"Expected 2 {EMAIL}"]
        },
    )
    return worker._build_ui_result(
        req={"metrics": None},
        framework="generic",
        queries=[f"Query one {EMAIL}", f"Query two {EMAIL}"],
        traces=[session],
        ground_truths=[ground_truth],
        results=[_stub_result()],
        errors=[],
        sessions_meta=[
            {
                "session_id": "session_001",
                "queries": [f"Query one {EMAIL}", f"Query two {EMAIL}"],
            }
        ],
    )


def _single_turn_ui_result(worker) -> Dict[str, Any]:
    return worker._build_ui_result(
        req={"metrics": None},
        framework="generic",
        queries=[f"Query {EMAIL}"],
        traces=[_assistant_trace(f"Response {EMAIL}")],
        ground_truths=[GroundTruth(expected_output=f"Expected {EMAIL}")],
        results=[_stub_result()],
        errors=[],
    )


def _display_fields(row: Dict[str, Any]) -> Dict[str, str]:
    """Every user-visible string in a result row, flattened for assertion."""
    fields = {
        "query": str(row.get("query") or ""),
        "agent_response": str(row.get("agent_response") or ""),
        "expected_answer": str(row.get("expected_answer") or ""),
    }
    for turn in row.get("turns") or []:
        prefix = f"turns[{turn.get('turn')}]"
        fields[f"{prefix}.query"] = str(turn.get("query") or "")
        fields[f"{prefix}.response"] = str(turn.get("response") or "")
        fields[f"{prefix}.expected"] = str(turn.get("expected") or "")
    return fields


class TestPiiRedactionDisabledByDefault:
    """Property 1: the control is inert unless a deployer opts in."""

    def test_single_turn_unredacted(self, monkeypatch):
        worker = _load_worker(monkeypatch, redact=False)
        row = _single_turn_ui_result(worker)["rows"][0]
        assert EMAIL in row["agent_response"]
        assert EMAIL in row["expected_answer"]

    def test_multi_turn_unredacted(self, monkeypatch):
        worker = _load_worker(monkeypatch, redact=False)
        row = _multi_turn_ui_result(worker)["rows"][0]
        fields = _display_fields(row)
        # Sanity: the fixture really does put PII in every field the redacted
        # case asserts on, so that test cannot pass vacuously.
        carrying = {k for k, v in fields.items() if EMAIL in v}
        assert carrying == set(fields) - {"query"}, (
            f"fixture did not populate every field: {sorted(set(fields) - carrying)}"
        )


class TestPiiRedactionEnabled:
    """Property 2: nothing user-visible survives with the email intact."""

    def test_single_turn_redacted(self, monkeypatch):
        worker = _load_worker(monkeypatch, redact=True)
        row = _single_turn_ui_result(worker)["rows"][0]
        leaked = sorted(k for k, v in _display_fields(row).items() if EMAIL in v)
        assert leaked == [], f"PII leaked into: {leaked}"

    def test_multi_turn_redacted_including_per_turn_fields(self, monkeypatch):
        worker = _load_worker(monkeypatch, redact=True)
        row = _multi_turn_ui_result(worker)["rows"][0]

        # The per-turn breakdown must actually be present, or the leak check
        # below would pass simply because there is nothing to inspect.
        assert len(row["turns"]) == 2
        assert row["turn_count"] == 2

        fields = _display_fields(row)
        assert len(fields) == 9  # 3 conversation-level + 3 per turn * 2 turns
        leaked = sorted(k for k, v in fields.items() if EMAIL in v)
        assert leaked == [], f"PII leaked into: {leaked}"

    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes"])
    def test_truthy_env_values_enable_redaction(self, monkeypatch, value):
        monkeypatch.setenv("UAEF_REDACT_PII_ON_PERSIST", value)
        sys.modules.pop("handlers.worker", None)
        worker = importlib.import_module("handlers.worker")
        assert worker._pii_redaction_enabled() is True

    def test_session_label_is_not_user_text(self, monkeypatch):
        """The multi-turn label is a session id + turn count, not user input."""
        worker = _load_worker(monkeypatch, redact=True)
        row = _multi_turn_ui_result(worker)["rows"][0]
        assert row["query"] == "session_001 (2 turns)"
        assert row["session_id"] == "session_001"
