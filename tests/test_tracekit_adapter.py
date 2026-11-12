# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Tracekit adapter.

``data/tracekit/run.tkb`` is a bundle exported by tracekit-ai 1.6.0 from the scripted run in
``data/tracekit/make_bundle.py``: a support agent's six tool calls under a small policy, with
a person answering the two held refunds. ``trust.json`` pins the key of the signer that
signed it. Tests that read the bundle need the 'tracekit' extra and skip without it.
"""

import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

from uaef.adapters import get_adapter
from uaef.adapters.base import AdapterTransformationError
from uaef.adapters.tracekit import TracekitAdapter
from uaef.api import evaluate
from uaef.models.agent_trace import AgentTrace
from uaef.models.ground_truth import GroundTruth
from uaef.models.tool_call import ToolCall

DATA = Path(__file__).parent / "data" / "tracekit"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw():
    pytest.importorskip("tracekit")
    return {"bundle_path": str(DATA / "run.tkb"), "trust_path": str(DATA / "trust.json")}


@pytest.fixture
def tampered(raw, tmp_path):
    """The bundle with the denied delete_account call's decision edited to allow.

    The manifest's hashes are refreshed, as someone editing the run's file would do; only
    the record's signature can tell.
    """
    with zipfile.ZipFile(raw["bundle_path"]) as z:
        files = {name: z.read(name) for name in z.namelist()}
    run = next(name for name in files if name.startswith("runs/"))
    assert files[run].count(b'"decision":"deny"') == 1
    files[run] = files[run].replace(b'"decision":"deny"', b'"decision":"allow"')
    manifest = json.loads(files["manifest.json"])
    manifest["files"][run] = hashlib.sha256(files[run]).hexdigest()
    files["manifest.json"] = json.dumps(manifest).encode()
    path = tmp_path / "tampered.tkb"
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return {**raw, "bundle_path": str(path)}


# ---------------------------------------------------------------------------
# Verified bundle
# ---------------------------------------------------------------------------


class TestVerifiedBundle:
    def test_returns_agent_trace(self, raw):
        trace = TracekitAdapter().transform_to_canonical(raw)
        assert isinstance(trace, AgentTrace)
        assert trace.framework == "tracekit"
        assert trace.trace_id == UUID(trace.metadata["tracekit"]["run_id"])
        assert trace.session_id == trace.metadata["tracekit"]["run_id"]
        assert trace.metadata["tracekit"]["agent"] == {"name": "support-agent"}

    def test_verification_in_metadata(self, raw):
        verification = (
            TracekitAdapter().transform_to_canonical(raw).metadata["tracekit"]["verification"]
        )
        assert verification["verified"] is True
        assert verification["integrity"] == "VERIFIED"
        assert verification["reason"].startswith("VERIFIED: 6 tool calls in 18 signed records")
        assert any(line.startswith("PROOF AGAINST THE AGENT") for line in verification["summary"])
        assert verification["verifier"].startswith("tracekit-ai ")

    def test_tool_calls_in_run_order(self, raw):
        trace = TracekitAdapter().transform_to_canonical(raw)
        assert [tc.name for tc in trace.tool_calls] == [
            "lookup_order",
            "send_email",
            "issue_refund",
            "issue_refund",
            "delete_account",
            "lookup_order",
        ]
        timestamps = [tc.timestamp for tc in trace.tool_calls]
        assert timestamps == sorted(timestamps)
        assert all(tc.timestamp.tzinfo == timezone.utc for tc in trace.tool_calls)

    def test_no_invented_values(self, raw):
        """The bundle commits to arguments, results and text; none of them is filled in."""
        trace = TracekitAdapter().transform_to_canonical(raw)
        assert trace.messages == []
        assert all(tc.arguments == {} for tc in trace.tool_calls)
        assert all(tc.result is None and tc.error is None for tc in trace.tool_calls)
        assert trace.input_tokens is None and trace.cost is None and trace.latency is None

    def test_policy_decisions_labelled(self, raw):
        calls = TracekitAdapter().transform_to_canonical(raw).metadata["tracekit"]["calls"]
        assert [(c["decision"], c["rule_ids"]) for c in calls] == [
            ("allow", []),
            ("flag", ["SHOP-F001"]),
            ("ask", ["SHOP-A001"]),
            ("ask", ["SHOP-A001"]),
            ("deny", ["SHOP-D001"]),
            ("allow", []),
        ]
        assert [c["tool_use_id"] for c in calls] == [f"call-{i}" for i in range(1, 7)]
        assert all(c["args_commitment"].startswith("hmac-sha256:") for c in calls)

    def test_person_answers_labelled(self, raw):
        calls = TracekitAdapter().transform_to_canonical(raw).metadata["tracekit"]["calls"]
        assert calls[2]["approval"]["decision"] == "approve"
        assert calls[3]["approval"]["decision"] == "reject"
        for answered in calls[2:4]:
            assert answered["approval"]["person"] == "alice"
            assert answered["approval"]["self_approved"] is False
        assert all("approval" not in c for c in calls[:2] + calls[4:])

    def test_results_labelled(self, raw):
        calls = TracekitAdapter().transform_to_canonical(raw).metadata["tracekit"]["calls"]
        assert [c.get("result", {}).get("ok") for c in calls] == [
            True,
            True,
            True,
            None,  # rejected: never ran
            None,  # denied: never ran
            False,  # the tool failed
        ]
        assert calls[0]["result"]["output"]["hash"].startswith("hmac-sha256:")

    def test_labels_cite_signed_records(self, raw):
        calls = TracekitAdapter().transform_to_canonical(raw).metadata["tracekit"]["calls"]
        assert [c["record"]["run_seq"] for c in calls] == sorted(
            c["record"]["run_seq"] for c in calls
        )
        assert all(c["record"]["hash"].startswith("sha256:") for c in calls)

    def test_registered_and_evaluated_by_name(self, raw):
        """evaluate(adapter="tracekit") scores the run with deterministic metrics, no AWS."""
        assert isinstance(get_adapter("tracekit"), TracekitAdapter)
        now = datetime.now(timezone.utc)
        ground_truth = GroundTruth(
            expected_tool_calls=[
                ToolCall(name=name, timestamp=now)
                for name in ("lookup_order", "send_email", "issue_refund")
            ]
        )
        result = evaluate(
            trace=raw,
            adapter="tracekit",
            ground_truth=ground_truth,
            metrics=["tool_selection_accuracy", "tool_sequence_correctness"],
        )
        scores = {m.metric_name: m.score for d in result.dimension_results for m in d.metric_scores}
        assert scores == {"tool_selection_accuracy": 0.5, "tool_sequence_correctness": 0.5}


# ---------------------------------------------------------------------------
# Bundles that do not verify
# ---------------------------------------------------------------------------


class TestUnverifiedBundle:
    def test_tampered_bundle_raises(self, tampered):
        with pytest.raises(AdapterTransformationError, match="did not verify: TAMPERED at event"):
            TracekitAdapter().transform_to_canonical(tampered)

    def test_tampered_bundle_mapped_on_opt_out(self, tampered):
        trace = TracekitAdapter(allow_unverified=True).transform_to_canonical(tampered)
        verification = trace.metadata["tracekit"]["verification"]
        assert verification["verified"] is False
        assert verification["integrity"] == "FAILED"
        assert verification["reason"].startswith("TAMPERED at event")
        # what the edited record claims, flagged unverified above
        assert trace.metadata["tracekit"]["calls"][4]["decision"] == "allow"

    def test_opt_out_through_registry_config(self, tampered):
        adapter = get_adapter("tracekit", {"allow_unverified": True})
        trace = adapter.transform_to_canonical(tampered)
        assert trace.metadata["tracekit"]["verification"]["verified"] is False

    def test_missing_trust_file_raises(self, raw, tmp_path):
        with pytest.raises(AdapterTransformationError, match="UNVERIFIABLE"):
            TracekitAdapter().transform_to_canonical(
                {**raw, "trust_path": str(tmp_path / "missing.json")}
            )

    def test_unreadable_bundle_raises_on_opt_out(self, raw, tmp_path):
        """Nothing to map: the opt-out does not turn an unreadable file into an empty trace."""
        path = tmp_path / "not-a-bundle.tkb"
        path.write_bytes(b"not a zip")
        with pytest.raises(AdapterTransformationError, match="found 0"):
            TracekitAdapter(allow_unverified=True).transform_to_canonical(
                {**raw, "bundle_path": str(path)}
            )


# ---------------------------------------------------------------------------
# Input and install errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_required_keys_raises(self):
        with pytest.raises(AdapterTransformationError, match="bundle_path, trust_path"):
            TracekitAdapter().transform_to_canonical({})

    def test_missing_extra_raises_actionable_import_error(self):
        with patch.dict(sys.modules, {"tracekit": None}):
            with pytest.raises(ImportError, match=r"pip install 'uaef\[tracekit\]'"):
                TracekitAdapter().transform_to_canonical(
                    {"bundle_path": "run.tkb", "trust_path": "trust.json"}
                )
