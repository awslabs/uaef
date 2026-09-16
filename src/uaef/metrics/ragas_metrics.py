# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""RAGAS integration metrics for UAEF metric registry.

Wraps RAGAS connector methods as BaseMetric subclasses so they can be
registered in the MetricRegistry alongside built-in metrics.

To add a new RAGAS metric:
1. Create a subclass with _ragas_import, _ragas_result_key, and optional
   _requires_context / _requires_ground_truth_output / _uses_multi_turn.
2. Implement get_name(), requires_ground_truth(), requires_llm_judge(), calculate().
3. Add batch_group() and batch_calculate() (copy from existing classes).
The metric will automatically participate in batched evaluation.
"""

import math
import threading
from typing import Any, Dict, List, Optional

from uaef.metrics.base import BaseMetric
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.metric_score import MetricScore
from uaef.logging import get_logger
from uaef.integrations._guards import integration_missing_error

logger = get_logger(__name__)

_RAGAS_FEATURE = "RAGAS metrics"

_cached_connector = None
_connector_lock = threading.Lock()


def _get_connector(llm=None, embeddings=None):
    global _cached_connector
    if _cached_connector is not None:
        return _cached_connector
    with _connector_lock:
        if _cached_connector is not None:
            return _cached_connector
        from uaef.integrations.ragas_connector import RAGASConnector
        connector = RAGASConnector(llm=llm, embeddings=embeddings)
        if llm is None and embeddings is None:
            _cached_connector = connector
        return connector


# Module-level overrides for custom LLM/embeddings.
ragas_llm = None
ragas_embeddings = None

_RAGAS_BATCH_GROUP = "ragas"


def _import_ragas_metric(import_path: str) -> Any:
    """Dynamically import a RAGAS metric object from a dotted path."""
    module_path, attr_name = import_path.rsplit(".", 1)
    import importlib
    # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import -- False positive: import_path is a library-defined RAGAS dotted path (internal constant), never user-controlled input.
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _ragas_batch_calculate(
    metrics: List[BaseMetric],
    evaluation_input: EvaluationInput,
) -> List[MetricScore]:
    """Evaluate multiple RAGAS metrics in a single ragas.evaluate() call.

    Reads _ragas_import, _ragas_result_key, _requires_context,
    _requires_ground_truth_output, and _uses_multi_turn from each metric
    class to decide eligibility. New metrics just set those attributes.
    """
    connector = _get_connector(llm=ragas_llm, embeddings=ragas_embeddings)
    if not connector.is_available():
        raise integration_missing_error(_RAGAS_FEATURE)

    try:
        from ragas import evaluate as ragas_evaluate
        from datasets import Dataset
    except ImportError:
        raise integration_missing_error(_RAGAS_FEATURE)

    ragas_data = connector.transform_to_ragas_format(evaluation_input)

    ragas_metric_objects = []
    metric_name_map: Dict[str, str] = {}
    fallback_scores: List[MetricScore] = []

    for metric in metrics:
        name = metric.get_name()
        ragas_import = getattr(metric, '_ragas_import', None)
        result_key = getattr(metric, '_ragas_result_key', None)
        requires_ctx = getattr(metric, '_requires_context', False)
        requires_gt = getattr(metric, '_requires_ground_truth_output', False)
        uses_mt = getattr(metric, '_uses_multi_turn', False)

        if uses_mt or ragas_import is None:
            try:
                score = metric.calculate(evaluation_input)
                fallback_scores.append(score)
            except Exception as e:
                logger.error(f"Error calculating metric {name}: {e}")
            continue

        if requires_ctx and not evaluation_input.context:
            logger.error(f"Error calculating metric {name}: Context is required")
            continue
        if requires_gt and (
            not evaluation_input.ground_truth
            or not evaluation_input.ground_truth.expected_output
        ):
            logger.error(
                f"Error calculating metric {name}: Ground truth expected_output is required"
            )
            continue

        try:
            ragas_obj = _import_ragas_metric(ragas_import)
            ragas_metric_objects.append(ragas_obj)
            metric_name_map[result_key] = name
        except Exception as e:
            logger.error(f"Error importing RAGAS metric for {name}: {e}")

    if not ragas_metric_objects:
        return fallback_scores

    # Deduplicate (multiple UAEF metrics may use same RAGAS metric object)
    seen: Dict[str, Any] = {}
    unique = []
    for rm in ragas_metric_objects:
        key = getattr(rm, 'name', str(rm))
        if key not in seen:
            seen[key] = rm
            unique.append(rm)
    ragas_metric_objects = unique

    dataset_dict: Dict[str, list] = {
        "question": [ragas_data["question"]],
        "answer": [ragas_data["answer"]],
    }
    if ragas_data.get("contexts"):
        dataset_dict["contexts"] = [ragas_data["contexts"]]
    else:
        dataset_dict["contexts"] = [[]]
    if ragas_data.get("ground_truth"):
        dataset_dict["ground_truth"] = [ragas_data["ground_truth"]]

    dataset = Dataset.from_dict(dataset_dict)

    # RAGAS requires llm/embeddings to be set on metric objects directly
    # when using batch evaluation. The evaluate() kwargs don't always propagate.
    eval_kwargs = connector._evaluate_kwargs()
    llm_instance = eval_kwargs.get("llm")
    embeddings_instance = eval_kwargs.get("embeddings")
    for rm in ragas_metric_objects:
        if llm_instance and hasattr(rm, 'llm'):
            rm.llm = llm_instance
        if embeddings_instance and hasattr(rm, 'embeddings'):
            rm.embeddings = embeddings_instance
        # Some metrics have sub-metrics (e.g., answer_correctness has answer_similarity)
        if hasattr(rm, 'answer_similarity') and rm.answer_similarity is not None:
            if llm_instance and hasattr(rm.answer_similarity, 'llm'):
                rm.answer_similarity.llm = llm_instance
            if embeddings_instance and hasattr(rm.answer_similarity, 'embeddings'):
                rm.answer_similarity.embeddings = embeddings_instance

    result = ragas_evaluate(dataset, metrics=ragas_metric_objects, **eval_kwargs)

    batch_scores: List[MetricScore] = []
    for rkey, uaef_name in metric_name_map.items():
        try:
            score_val = connector._extract_score(result, rkey)
            if math.isnan(score_val):
                logger.error(f"Error calculating metric {uaef_name}: Score is nan, skipping")
                continue
            batch_scores.append(MetricScore(
                metric_name=uaef_name,
                score=float(score_val),
                reasoning=f"RAGAS {rkey} (batched evaluation)",
                metadata={"library": "ragas", "metric": rkey, "batched": True},
            ))
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error extracting RAGAS score for {uaef_name}: {e}")

    return batch_scores + fallback_scores


# ---------------------------------------------------------------------------
# Metric classes. Each declares its RAGAS import path and result key so
# _ragas_batch_calculate can handle it generically.
# ---------------------------------------------------------------------------


class RagasFaithfulnessMetric(BaseMetric):
    _ragas_import = "ragas.metrics.faithfulness"
    _ragas_result_key = "faithfulness"
    _requires_context = True

    def get_name(self): return "ragas_faithfulness"
    def requires_ground_truth(self): return False
    def requires_llm_judge(self): return True
    def get_dimension(self): return "RAGAS"
    def get_description(self): return "Measures if every claim in the answer is supported by retrieved context (RAGAS)"
    def batch_group(self): return _RAGAS_BATCH_GROUP

    @classmethod
    def batch_calculate(cls, metrics, evaluation_input):
        return _ragas_batch_calculate(metrics, evaluation_input)

    def calculate(self, evaluation_input):
        c = _get_connector(llm=ragas_llm, embeddings=ragas_embeddings)
        if not c.is_available(): raise integration_missing_error(_RAGAS_FEATURE)
        return c.calculate_faithfulness(evaluation_input)

    async def calculate_async(self, evaluation_input):
        return self.calculate(evaluation_input)


class RagasContextPrecisionMetric(BaseMetric):
    _ragas_import = "ragas.metrics.context_precision"
    _ragas_result_key = "context_precision"
    _requires_context = True
    _requires_ground_truth_output = True

    def get_name(self): return "ragas_context_precision"
    def requires_ground_truth(self): return True
    def requires_llm_judge(self): return True
    def get_dimension(self): return "RAGAS"
    def get_description(self): return "Measures if top-ranked retrieved chunks are the most relevant (RAGAS)"
    def batch_group(self): return _RAGAS_BATCH_GROUP

    @classmethod
    def batch_calculate(cls, metrics, evaluation_input):
        return _ragas_batch_calculate(metrics, evaluation_input)

    def calculate(self, evaluation_input):
        c = _get_connector(llm=ragas_llm, embeddings=ragas_embeddings)
        if not c.is_available(): raise integration_missing_error(_RAGAS_FEATURE)
        return c.calculate_context_precision(evaluation_input)

    async def calculate_async(self, evaluation_input):
        return self.calculate(evaluation_input)


class RagasContextRecallMetric(BaseMetric):
    _ragas_import = "ragas.metrics.context_recall"
    _ragas_result_key = "context_recall"
    _requires_context = True
    _requires_ground_truth_output = True

    def get_name(self): return "ragas_context_recall"
    def requires_ground_truth(self): return True
    def requires_llm_judge(self): return True
    def get_dimension(self): return "RAGAS"
    def get_description(self): return "Measures coverage of ground truth claims in retrieved context (RAGAS)"
    def batch_group(self): return _RAGAS_BATCH_GROUP

    @classmethod
    def batch_calculate(cls, metrics, evaluation_input):
        return _ragas_batch_calculate(metrics, evaluation_input)

    def calculate(self, evaluation_input):
        c = _get_connector(llm=ragas_llm, embeddings=ragas_embeddings)
        if not c.is_available(): raise integration_missing_error(_RAGAS_FEATURE)
        return c.calculate_context_recall(evaluation_input)

    async def calculate_async(self, evaluation_input):
        return self.calculate(evaluation_input)


class RagasAnswerPrecisionMetric(BaseMetric):
    _ragas_import = "ragas.metrics.answer_relevancy"
    _ragas_result_key = "answer_relevancy"

    def get_name(self): return "ragas_answer_precision"
    def requires_ground_truth(self): return False
    def requires_llm_judge(self): return True
    def get_dimension(self): return "RAGAS"
    def get_description(self): return "Measures relevance and support of the answer from context (RAGAS)"
    def batch_group(self): return _RAGAS_BATCH_GROUP

    @classmethod
    def batch_calculate(cls, metrics, evaluation_input):
        return _ragas_batch_calculate(metrics, evaluation_input)

    def calculate(self, evaluation_input):
        c = _get_connector(llm=ragas_llm, embeddings=ragas_embeddings)
        if not c.is_available(): raise integration_missing_error(_RAGAS_FEATURE)
        return c.calculate_answer_precision(evaluation_input)

    async def calculate_async(self, evaluation_input):
        return self.calculate(evaluation_input)


class RagasAnswerRecallMetric(BaseMetric):
    _ragas_import = "ragas.metrics.answer_correctness"
    _ragas_result_key = "answer_correctness"
    _requires_ground_truth_output = True

    def get_name(self): return "ragas_answer_recall"
    def requires_ground_truth(self): return True
    def requires_llm_judge(self): return True
    def get_dimension(self): return "RAGAS"
    def get_description(self): return "Measures coverage of ground truth information in the answer (RAGAS)"
    def batch_group(self): return _RAGAS_BATCH_GROUP

    @classmethod
    def batch_calculate(cls, metrics, evaluation_input):
        return _ragas_batch_calculate(metrics, evaluation_input)

    def calculate(self, evaluation_input):
        c = _get_connector(llm=ragas_llm, embeddings=ragas_embeddings)
        if not c.is_available(): raise integration_missing_error(_RAGAS_FEATURE)
        return c.calculate_answer_recall(evaluation_input)

    async def calculate_async(self, evaluation_input):
        return self.calculate(evaluation_input)


class RagasAnswerCorrectnessMetric(BaseMetric):
    _ragas_import = "ragas.metrics.answer_correctness"
    _ragas_result_key = "answer_correctness"
    _requires_ground_truth_output = True

    def get_name(self): return "ragas_answer_correctness"
    def requires_ground_truth(self): return True
    def requires_llm_judge(self): return True
    def get_dimension(self): return "RAGAS"
    def get_description(self): return "Combines factual accuracy and semantic similarity with ground truth (RAGAS)"
    def batch_group(self): return _RAGAS_BATCH_GROUP

    @classmethod
    def batch_calculate(cls, metrics, evaluation_input):
        return _ragas_batch_calculate(metrics, evaluation_input)

    def calculate(self, evaluation_input):
        c = _get_connector(llm=ragas_llm, embeddings=ragas_embeddings)
        if not c.is_available(): raise integration_missing_error(_RAGAS_FEATURE)
        return c.calculate_answer_correctness(evaluation_input)

    async def calculate_async(self, evaluation_input):
        return self.calculate(evaluation_input)


class RagasToolCallAccuracyMetric(BaseMetric):
    _ragas_import = "ragas.metrics.ToolCallAccuracy"
    _ragas_result_key = "tool_call_accuracy"
    _uses_multi_turn = True  # Uses MultiTurnSample, can't batch with Dataset metrics

    def get_name(self): return "ragas_tool_call_accuracy"
    def requires_ground_truth(self): return True
    def requires_llm_judge(self): return False
    def get_dimension(self): return "RAGAS"
    def get_description(self): return "Measures correctness of tool selection and usage vs expected (RAGAS)"
    def batch_group(self): return _RAGAS_BATCH_GROUP

    @classmethod
    def batch_calculate(cls, metrics, evaluation_input):
        return _ragas_batch_calculate(metrics, evaluation_input)

    def calculate(self, evaluation_input):
        c = _get_connector(llm=ragas_llm, embeddings=ragas_embeddings)
        if not c.is_available(): raise integration_missing_error(_RAGAS_FEATURE)
        return c.calculate_tool_call_accuracy(evaluation_input)

    async def calculate_async(self, evaluation_input):
        return self.calculate(evaluation_input)
