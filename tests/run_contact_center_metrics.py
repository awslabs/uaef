#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end demo for the contact-center (use-case-specific) metrics.

This script exercises ``ContainmentMetric`` and ``ResolutionMetric`` from
``src/uaef/metrics/use_case_specific.py``. It is NOT a pytest test — it can
make real Bedrock API calls.

Three modes (pick with --mode):
  smoke   — registry visibility + class instantiation, no Bedrock calls
  mocked  — full evaluate() pipeline with the Bedrock client patched out
            so it returns canned JSON. No AWS creds needed.
  live    — full evaluate() pipeline against real Bedrock. Requires AWS
            creds with bedrock-runtime invoke_model permission and the
            judge model id resolvable via uaef.config.get_config().

Usage:
    cd agenticevaluationframework
    uv run python tests/run_contact_center_metrics.py --mode smoke
    uv run python tests/run_contact_center_metrics.py --mode mocked
    uv run python tests/run_contact_center_metrics.py --mode live
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch
from uuid import uuid4

# Make sure ``src`` is importable when run from the repo root.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _build_traces() -> tuple[object, object]:
    """Build two short customer-service conversations that contrast on
    containment and resolution outcomes."""
    from uaef.models import AgentTrace
    from uaef.models.message import Message, MessageRole

    def turn(role: MessageRole, content: str) -> Message:
        return Message(role=role, content=content, timestamp=datetime.now(timezone.utc))

    contained_resolved = AgentTrace(
        trace_id=uuid4(),
        messages=[
            turn(MessageRole.USER,
                 "Hi, I'm Alice. I'd like to cancel order #12345."),
            turn(MessageRole.ASSISTANT,
                 "Hi Alice! I can help. Could you confirm your email?"),
            turn(MessageRole.USER, "alice@example.com"),
            turn(MessageRole.ASSISTANT,
                 "Thanks. I've cancelled order #12345 and emailed the refund details. "
                 "You'll see the refund in 3-5 business days."),
            turn(MessageRole.USER, "Great, thank you!"),
            turn(MessageRole.ASSISTANT,
                 "You're welcome. Is there anything else?"),
        ],
        tool_calls=[],
    )

    escalated_unresolved = AgentTrace(
        trace_id=uuid4(),
        messages=[
            turn(MessageRole.USER,
                 "I want a refund on order #99999. The product was broken."),
            turn(MessageRole.ASSISTANT,
                 "I'm sorry to hear that. Unfortunately I can't issue refunds — "
                 "please contact our human support team at support@example.com or "
                 "call our helpline. They'll be able to process this for you."),
            turn(MessageRole.USER, "Okay, I'll do that."),
        ],
        tool_calls=[],
    )

    return contained_resolved, escalated_unresolved


def _build_ground_truth() -> object:
    """A GroundTruth with the optional fields the AMACE-derived metrics read."""
    from uaef.models.ground_truth import GroundTruth

    return GroundTruth(
        expected_output="Resolve the customer's order issue.",
        expected_arguments={
            "chatbot_role": (
                "You are a customer support specialist for ACME Corp. Help with "
                "cancellations, refunds, and order questions. Always greet by name "
                "and confirm details before taking action."
            ),
            "system_instructions": (
                "Always greet the customer by name. Confirm the order number "
                "before any action. State the refund timeline clearly."
            ),
        },
    )


def _print_result(label: str, result) -> None:
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"Overall: {result.overall_score:.2f}   passed={result.passed}")
    for dim in result.dimension_results:
        print(f"\n{dim.dimension_name} ({dim.aggregate_score:.2f}):")
        for m in dim.metric_scores:
            score = f"{m.score:.2f}" if m.score is not None else "N/A"
            raw = m.metadata.get("raw") if m.metadata else None
            extra = f"  raw={raw}" if raw is not None else ""
            print(f"  {m.metric_name:<28} {score}{extra}")
            if m.reasoning:
                snippet = m.reasoning.replace("\n", " ").strip()[:140]
                print(f"    {snippet}")


# ---------------------------------------------------------------------------
# Mode: smoke
# ---------------------------------------------------------------------------


def run_smoke() -> int:
    """Confirm registration plumbing without invoking any judge."""
    from uaef.metrics import (
        ContainmentMetric,
        ResolutionMetric,
        list_metrics,
    )
    from uaef.metrics.registry import register_use_case_specific_metrics

    names = set(list_metrics())
    assert "containment" not in names, "Defaults must NOT include containment"
    assert "resolution" not in names, "Defaults must NOT include resolution"
    print(f"[smoke] Defaults exclude containment/resolution ({len(names)} default metrics)")

    # Direct class import (advanced usage)
    assert ContainmentMetric().get_name() == "containment"
    assert ContainmentMetric().get_dimension() == "Multi-Turn"
    assert ResolutionMetric().get_name() == "resolution"
    assert ResolutionMetric().get_dimension() == "Multi-Turn"
    print("[smoke] ContainmentMetric and ResolutionMetric importable from uaef.metrics")

    # Opt in
    register_use_case_specific_metrics()
    names_after = set(list_metrics())
    assert {"containment", "resolution"} <= names_after
    print(f"[smoke] After opt-in: {len(names_after)} metrics; containment/resolution present")

    # Idempotent
    register_use_case_specific_metrics()
    register_use_case_specific_metrics()
    print("[smoke] register_use_case_specific_metrics() is idempotent")

    print("\n[smoke] PASS")
    return 0


# ---------------------------------------------------------------------------
# Mode: mocked
# ---------------------------------------------------------------------------


def _make_mock_bedrock(responses: list[dict]) -> MagicMock:
    """Build a MagicMock for the boto3 bedrock-runtime client that returns
    canned Anthropic-style payloads in order. Reset the iterator each session."""
    iterator = iter(responses)

    def invoke_model(**kwargs) -> dict:
        body = next(iterator)
        text = json.dumps(body)
        # Mimic the bedrock-runtime response shape: a stream-like object with
        # ``read()`` returning bytes containing {"content":[{"text": "<json>"}]}
        wire_payload = {"content": [{"type": "text", "text": text}]}
        body_obj = MagicMock()
        body_obj.read.return_value = json.dumps(wire_payload).encode("utf-8")
        return {"body": body_obj}

    client = MagicMock()
    client.invoke_model.side_effect = invoke_model
    return client


def run_mocked() -> int:
    """Run the full evaluate() pipeline with Bedrock patched out."""
    from uaef.api import evaluate
    from uaef.metrics.registry import register_use_case_specific_metrics

    register_use_case_specific_metrics()

    contained_trace, escalated_trace = _build_traces()
    gt = _build_ground_truth()

    # Two judge calls per evaluate() — one for containment, one for resolution.
    # Order matches the registration order in register_use_case_specific_metrics
    # (Containment, Resolution).
    contained_responses = [
        {"Score": 1, "Reason": "Agent handled cancellation without escalating to a human."},
        {"Resolution": 1, "Reason": "Order was cancelled and refund details were emailed."},
    ]
    escalated_responses = [
        {"Score": 0, "Reason": "Agent redirected the user to support@example.com — escalation."},
        {"Resolution": 0, "Reason": "Customer was redirected to human support; issue was not resolved by the bot."},
    ]

    # Patch boto3.client for both evaluations. The metrics use
    # ``boto3.client("bedrock-runtime", ...)`` directly inside
    # ``_invoke_bedrock_judge``.
    def evaluate_with_mock(trace, mock_responses):
        with patch(
            "uaef.metrics.multi_turn.boto3.client",
            return_value=_make_mock_bedrock(mock_responses),
        ):
            return evaluate(
                trace=trace,
                ground_truth=gt,
                metrics=["containment", "resolution"],
            )

    r1 = evaluate_with_mock(contained_trace, contained_responses)
    _print_result("Mocked: contained + resolved conversation", r1)
    r2 = evaluate_with_mock(escalated_trace, escalated_responses)
    _print_result("Mocked: escalated + unresolved conversation", r2)

    # Hard assertions on the mocked outcomes
    def score_for(result, name) -> Optional[float]:
        for dim in result.dimension_results:
            for m in dim.metric_scores:
                if m.metric_name == name:
                    return m.score
        return None

    assert score_for(r1, "containment") == 1.0, "expected contained=1.0"
    assert score_for(r1, "resolution") == 1.0, "expected resolved=1.0"
    assert score_for(r2, "containment") == 0.0, "expected escalated=0.0"
    assert score_for(r2, "resolution") == 0.0, "expected unresolved=0.0"
    print("\n[mocked] Score assertions passed")
    print("[mocked] PASS")
    return 0


# ---------------------------------------------------------------------------
# Mode: live
# ---------------------------------------------------------------------------


def run_live() -> int:
    """Real Bedrock end-to-end. Requires creds + invoke_model permissions."""
    from uaef.api import evaluate
    from uaef.metrics.registry import register_use_case_specific_metrics

    register_use_case_specific_metrics()

    contained_trace, escalated_trace = _build_traces()
    gt = _build_ground_truth()

    metrics = ["containment", "resolution"]

    print("[live] Evaluating contained-resolved conversation against real Bedrock...")
    r1 = evaluate(trace=contained_trace, ground_truth=gt, metrics=metrics)
    _print_result("Live: contained + resolved conversation", r1)

    print("\n[live] Evaluating escalated-unresolved conversation against real Bedrock...")
    r2 = evaluate(trace=escalated_trace, ground_truth=gt, metrics=metrics)
    _print_result("Live: escalated + unresolved conversation", r2)

    # Sanity-check: scores must be 0/1/None
    for label, r in [("contained", r1), ("escalated", r2)]:
        for dim in r.dimension_results:
            for m in dim.metric_scores:
                if m.metric_name in ("containment", "resolution"):
                    if m.score is not None:
                        assert m.score in (0.0, 1.0), (
                            f"[live] {label}: {m.metric_name} returned non-boolean {m.score}"
                        )
    print("\n[live] PASS")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "mocked", "live"),
        default="smoke",
        help="smoke (no Bedrock), mocked (patched boto3), live (real Bedrock).",
    )
    args = parser.parse_args()

    if args.mode == "smoke":
        return run_smoke()
    if args.mode == "mocked":
        return run_mocked()
    if args.mode == "live":
        return run_live()
    return 1  # unreachable


if __name__ == "__main__":
    sys.exit(main())
