# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepEval integration metrics for UAEF metric registry.

Wraps DeepEval connector methods as BaseMetric subclasses so they can be
registered in the MetricRegistry alongside built-in metrics.
"""

from typing import Optional

from uaef.metrics.base import BaseMetric
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.metric_score import MetricScore
from uaef.logging import get_logger
from uaef.integrations._guards import integration_missing_error

logger = get_logger(__name__)

# Human-readable label used in the actionable ImportError raised when the
# DeepEval backend (shipped behind the 'integrations' extra) is not installed.
_DEEPEVAL_FEATURE = "DeepEval metrics"


def _get_connector():
    from uaef.integrations.deepeval_connector import DeepEvalConnector
    return DeepEvalConnector()


class DeepEvalContextualPrecisionMetric(BaseMetric):
    def get_name(self) -> str:
        return "deepeval_contextual_precision"

    def requires_ground_truth(self) -> bool:
        return True

    def requires_llm_judge(self) -> bool:
        return True

    def get_description(self) -> Optional[str]:
        return "Measures whether retrieved context is relevant and irrelevant context is ranked lower (DeepEval)"

    def get_dimension(self) -> Optional[str]:
        return "DeepEval"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        c = _get_connector()
        if not c.is_available():
            raise integration_missing_error(_DEEPEVAL_FEATURE)
        return c.calculate_contextual_precision(evaluation_input)

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        return self.calculate(evaluation_input)


class DeepEvalContextualRecallMetric(BaseMetric):
    def get_name(self) -> str:
        return "deepeval_contextual_recall"

    def requires_ground_truth(self) -> bool:
        return True

    def requires_llm_judge(self) -> bool:
        return True

    def get_description(self) -> Optional[str]:
        return "Measures whether all relevant information from expected output is in retrieved context (DeepEval)"

    def get_dimension(self) -> Optional[str]:
        return "DeepEval"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        c = _get_connector()
        if not c.is_available():
            raise integration_missing_error(_DEEPEVAL_FEATURE)
        return c.calculate_contextual_recall(evaluation_input)

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        return self.calculate(evaluation_input)


class DeepEvalContextualRelevancyMetric(BaseMetric):
    def get_name(self) -> str:
        return "deepeval_contextual_relevancy"

    def requires_ground_truth(self) -> bool:
        return False

    def requires_llm_judge(self) -> bool:
        return True

    def get_description(self) -> Optional[str]:
        return "Assesses whether retrieved context is relevant to the input query (DeepEval)"

    def get_dimension(self) -> Optional[str]:
        return "DeepEval"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        c = _get_connector()
        if not c.is_available():
            raise integration_missing_error(_DEEPEVAL_FEATURE)
        return c.calculate_contextual_relevancy(evaluation_input)

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        return self.calculate(evaluation_input)


class DeepEvalHallucinationMetric(BaseMetric):
    def get_name(self) -> str:
        return "deepeval_hallucination"

    def requires_ground_truth(self) -> bool:
        return False

    def requires_llm_judge(self) -> bool:
        return True

    def get_description(self) -> Optional[str]:
        return "Detects claims in the response not supported by context (DeepEval)"

    def get_dimension(self) -> Optional[str]:
        return "DeepEval"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        c = _get_connector()
        if not c.is_available():
            raise integration_missing_error(_DEEPEVAL_FEATURE)
        return c.calculate_hallucination(evaluation_input)

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        return self.calculate(evaluation_input)


class DeepEvalFaithfulnessMetric(BaseMetric):
    def get_name(self) -> str:
        return "deepeval_faithfulness"

    def requires_ground_truth(self) -> bool:
        return False

    def requires_llm_judge(self) -> bool:
        return True

    def get_description(self) -> Optional[str]:
        return "Measures whether the answer is factually consistent with retrieval context (DeepEval)"

    def get_dimension(self) -> Optional[str]:
        return "DeepEval"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        c = _get_connector()
        if not c.is_available():
            raise integration_missing_error(_DEEPEVAL_FEATURE)
        return c.calculate_faithfulness(evaluation_input)

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        return self.calculate(evaluation_input)


class DeepEvalAnswerRelevancyMetric(BaseMetric):
    def get_name(self) -> str:
        return "deepeval_answer_relevancy"

    def requires_ground_truth(self) -> bool:
        return False

    def requires_llm_judge(self) -> bool:
        return True

    def get_description(self) -> Optional[str]:
        return "Evaluates whether the generated answer is relevant to the query (DeepEval)"

    def get_dimension(self) -> Optional[str]:
        return "DeepEval"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        c = _get_connector()
        if not c.is_available():
            raise integration_missing_error(_DEEPEVAL_FEATURE)
        return c.calculate_answer_relevancy(evaluation_input)

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        return self.calculate(evaluation_input)


class DeepEvalToolCorrectnessMetric(BaseMetric):
    def get_name(self) -> str:
        return "deepeval_tool_correctness"

    def requires_ground_truth(self) -> bool:
        return True

    def requires_llm_judge(self) -> bool:
        return False

    def get_description(self) -> Optional[str]:
        return "Validates tool selection vs expected tool calls (DeepEval)"

    def get_dimension(self) -> Optional[str]:
        return "DeepEval"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        c = _get_connector()
        if not c.is_available():
            raise integration_missing_error(_DEEPEVAL_FEATURE)
        return c.calculate_tool_correctness(evaluation_input)

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        return self.calculate(evaluation_input)


class DeepEvalRoleAdherenceMetric(BaseMetric):
    def get_name(self) -> str:
        return "deepeval_role_adherence"

    def requires_ground_truth(self) -> bool:
        return True

    def requires_llm_judge(self) -> bool:
        return True

    def get_description(self) -> Optional[str]:
        return (
            "Evaluates whether the chatbot stays in its assigned role across "
            "the conversation, using DeepEval's RoleAdherenceMetric. "
            "Returns one MetricScore per evaluation; per-turn drift verdicts "
            "are surfaced in metadata['out_of_character_turns']."
        )

    def get_dimension(self) -> Optional[str]:
        return "DeepEval"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        c = _get_connector()
        if not c.is_available():
            raise integration_missing_error(_DEEPEVAL_FEATURE)
        return c.calculate_role_adherence(evaluation_input)

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        return self.calculate(evaluation_input)
