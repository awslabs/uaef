# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wrapper-level guard tests for the optional ``integrations`` extra.

Requirement 3.5: requesting an integration metric while its extra is not
installed raises an ImportError that names the extra and states the exact
``pip install 'uaef[<extra>]'`` command, and returns **no** metric result.

``tests/test_deepeval_connector.py`` and ``tests/test_ragas_connector.py``
already cover that contract at the *connector* level. These tests pin it at
the *metric wrapper* level — the layer the registry and evaluator actually
call — for every DeepEval and RAGAS wrapper, so a wrapper can't regress to
returning a degraded ``MetricScore`` (score 0.0) and have the failure silently
land in a report as a real evaluation result.

No live library, Bedrock, or judge calls: the connector factory is patched so
``is_available()`` returns False.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from uaef.metrics import deepeval_metrics, ragas_metrics
from uaef.models.agent_trace import AgentTrace
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.ground_truth import GroundTruth
from uaef.models.message import Message, MessageRole

# Matches the message built by uaef.integrations._guards.integration_missing_error:
#   "<feature> require the 'integrations' extra. Install with: pip install 'uaef[integrations]'"
_MISSING_EXTRA_RE = r"integrations.*pip install 'uaef\[integrations\]'"


# -- Fixtures / helpers ----------------------------------------------------


def _msg(role: MessageRole, content: str) -> Message:
    return Message(role=role, content=content, timestamp=datetime.now(timezone.utc))


def _trace(messages: List[Message]) -> AgentTrace:
    return AgentTrace(trace_id=uuid4(), messages=messages, tool_calls=[])


def _eval_input() -> EvaluationInput:
    """A generically-valid input, so any raise is the guard and not validation."""
    return EvaluationInput(
        trace=_trace(
            [
                _msg(MessageRole.USER, "u1"),
                _msg(MessageRole.ASSISTANT, "a1"),
            ]
        ),
        ground_truth=GroundTruth(
            expected_output="expected",
            expected_arguments={"chatbot_role": "ACME"},
        ),
        context=["some retrieved context"],
    )


def _unavailable_connector() -> MagicMock:
    """A connector mock that reports the backend as not installed."""
    connector = MagicMock()
    connector.is_available.return_value = False
    return connector


def _assert_no_metric_computed(connector: MagicMock) -> None:
    """Requirement 3.5: no metric result is produced when the guard trips.

    Every connector entry point the wrappers delegate to is named
    ``calculate_*`` / ``transform_*``; none may be reached.
    """
    called = [
        name
        for name, *_ in connector.method_calls
        if name.startswith("calculate_") or name.startswith("transform_")
    ]
    assert called == [], f"connector work attempted despite guard: {called}"


# -- DeepEval wrappers -----------------------------------------------------

# Every DeepEval wrapper registered in the metric registry.
DEEPEVAL_WRAPPERS = [
    (deepeval_metrics.DeepEvalContextualPrecisionMetric, "deepeval_contextual_precision"),
    (deepeval_metrics.DeepEvalContextualRecallMetric, "deepeval_contextual_recall"),
    (deepeval_metrics.DeepEvalContextualRelevancyMetric, "deepeval_contextual_relevancy"),
    (deepeval_metrics.DeepEvalHallucinationMetric, "deepeval_hallucination"),
    (deepeval_metrics.DeepEvalFaithfulnessMetric, "deepeval_faithfulness"),
    (deepeval_metrics.DeepEvalAnswerRelevancyMetric, "deepeval_answer_relevancy"),
    (deepeval_metrics.DeepEvalToolCorrectnessMetric, "deepeval_tool_correctness"),
    (deepeval_metrics.DeepEvalRoleAdherenceMetric, "deepeval_role_adherence"),
]

RAGAS_WRAPPERS = [
    (ragas_metrics.RagasFaithfulnessMetric, "ragas_faithfulness"),
    (ragas_metrics.RagasContextPrecisionMetric, "ragas_context_precision"),
    (ragas_metrics.RagasContextRecallMetric, "ragas_context_recall"),
    (ragas_metrics.RagasAnswerPrecisionMetric, "ragas_answer_precision"),
    (ragas_metrics.RagasAnswerRecallMetric, "ragas_answer_recall"),
    (ragas_metrics.RagasAnswerCorrectnessMetric, "ragas_answer_correctness"),
    (ragas_metrics.RagasToolCallAccuracyMetric, "ragas_tool_call_accuracy"),
]


class TestDeepEvalWrapperGuards:
    def test_wrapper_list_covers_every_registered_deepeval_metric(self):
        """Guard against a new wrapper being added without guard coverage."""
        from uaef.metrics.registry import list_metrics, reset_registry

        reset_registry()
        registered = {n for n in list_metrics() if n.startswith("deepeval_")}
        covered = {name for _, name in DEEPEVAL_WRAPPERS}
        assert registered == covered, (
            "DEEPEVAL_WRAPPERS is out of sync with the registry; "
            f"uncovered={registered - covered}, stale={covered - registered}"
        )

    @pytest.mark.parametrize(
        "metric_cls,metric_name",
        DEEPEVAL_WRAPPERS,
        ids=[name for _, name in DEEPEVAL_WRAPPERS],
    )
    def test_calculate_raises_when_unavailable(self, metric_cls, metric_name):
        metric = metric_cls()
        assert metric.get_name() == metric_name

        connector = _unavailable_connector()
        with patch.object(deepeval_metrics, "_get_connector", return_value=connector):
            with pytest.raises(ImportError, match=_MISSING_EXTRA_RE):
                metric.calculate(_eval_input())

        _assert_no_metric_computed(connector)

    @pytest.mark.parametrize(
        "metric_cls,metric_name",
        DEEPEVAL_WRAPPERS,
        ids=[name for _, name in DEEPEVAL_WRAPPERS],
    )
    async def test_calculate_async_raises_when_unavailable(self, metric_cls, metric_name):
        """The async entry point delegates to calculate() and must not swallow."""
        metric = metric_cls()
        connector = _unavailable_connector()
        with patch.object(deepeval_metrics, "_get_connector", return_value=connector):
            with pytest.raises(ImportError, match=_MISSING_EXTRA_RE):
                await metric.calculate_async(_eval_input())

        _assert_no_metric_computed(connector)


# -- RAGAS wrappers --------------------------------------------------------


class TestRagasWrapperGuards:
    def test_wrapper_list_covers_every_registered_ragas_metric(self):
        from uaef.metrics.registry import list_metrics, reset_registry

        reset_registry()
        registered = {n for n in list_metrics() if n.startswith("ragas_")}
        covered = {name for _, name in RAGAS_WRAPPERS}
        assert registered == covered, (
            "RAGAS_WRAPPERS is out of sync with the registry; "
            f"uncovered={registered - covered}, stale={covered - registered}"
        )

    @pytest.mark.parametrize(
        "metric_cls,metric_name",
        RAGAS_WRAPPERS,
        ids=[name for _, name in RAGAS_WRAPPERS],
    )
    def test_calculate_raises_when_unavailable(self, metric_cls, metric_name):
        metric = metric_cls()
        assert metric.get_name() == metric_name

        connector = _unavailable_connector()
        # ragas_metrics._get_connector takes (llm=, embeddings=); a MagicMock
        # with return_value accepts any signature.
        with patch.object(ragas_metrics, "_get_connector", return_value=connector):
            with pytest.raises(ImportError, match=_MISSING_EXTRA_RE):
                metric.calculate(_eval_input())

        _assert_no_metric_computed(connector)

    @pytest.mark.parametrize(
        "metric_cls,metric_name",
        RAGAS_WRAPPERS,
        ids=[name for _, name in RAGAS_WRAPPERS],
    )
    async def test_calculate_async_raises_when_unavailable(self, metric_cls, metric_name):
        metric = metric_cls()
        connector = _unavailable_connector()
        with patch.object(ragas_metrics, "_get_connector", return_value=connector):
            with pytest.raises(ImportError, match=_MISSING_EXTRA_RE):
                await metric.calculate_async(_eval_input())

        _assert_no_metric_computed(connector)

    def test_batch_calculate_raises_when_unavailable(self):
        """The batched RAGAS path shares the same guard.

        RAGAS metrics declare batch_group()=="ragas", so the evaluator routes
        them through _ragas_batch_calculate rather than per-metric calculate().
        That path must fail loud too, otherwise the guard is bypassed for the
        default (batched) execution route.
        """
        metrics = [cls() for cls, _ in RAGAS_WRAPPERS]
        connector = _unavailable_connector()
        with patch.object(ragas_metrics, "_get_connector", return_value=connector):
            with pytest.raises(ImportError, match=_MISSING_EXTRA_RE):
                ragas_metrics._ragas_batch_calculate(metrics, _eval_input())

        _assert_no_metric_computed(connector)

    def test_batch_calculate_via_public_classmethod_raises(self):
        """Same guard through the classmethod the evaluator actually calls."""
        metrics = [cls() for cls, _ in RAGAS_WRAPPERS]
        connector = _unavailable_connector()
        with patch.object(ragas_metrics, "_get_connector", return_value=connector):
            with pytest.raises(ImportError, match=_MISSING_EXTRA_RE):
                ragas_metrics.RagasFaithfulnessMetric.batch_calculate(
                    metrics, _eval_input()
                )

        _assert_no_metric_computed(connector)


# -- Error-message contract ------------------------------------------------


class TestGuardMessageContract:
    """The message must be actionable: name the extra and the pip command."""

    @pytest.mark.parametrize(
        "module,feature",
        [
            (deepeval_metrics, "DeepEval metrics"),
            (ragas_metrics, "RAGAS metrics"),
        ],
        ids=["deepeval", "ragas"],
    )
    def test_message_names_feature_extra_and_command(self, module, feature):
        from uaef.integrations._guards import integration_missing_error

        message = str(integration_missing_error(feature))
        assert feature in message
        assert "'integrations' extra" in message
        assert "pip install 'uaef[integrations]'" in message
