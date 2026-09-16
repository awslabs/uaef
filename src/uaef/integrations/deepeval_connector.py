# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepEval connector for external metric integration."""

import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

from uaef.models.evaluation_input import EvaluationInput
from uaef.models.metric_score import MetricScore
from uaef.logging import get_logger
from uaef.integrations._guards import integration_missing_error

# Human-readable label used in the actionable ImportError raised when the
# DeepEval backend (shipped behind the 'integrations' extra) is not installed.
_DEEPEVAL_FEATURE = "DeepEval metrics"

# DeepEval creates a `.deepeval` directory relative to os.getcwd() on import
# and during metric calculation. Lambda's root filesystem is read-only — only
# /tmp is writable. We must:
# 1. Set env vars DeepEval checks for its data directory.
# 2. Create the directory so os.makedirs doesn't fail.
# 3. Change the working directory to /tmp so any relative `.deepeval` writes
#    land in /tmp/.deepeval instead of failing on the read-only root.
_tmp_dir = tempfile.gettempdir()
_deepeval_dir = os.path.join(_tmp_dir, ".deepeval")
os.environ.setdefault("DEEPEVAL_RESULTS_FOLDER", _deepeval_dir)
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.makedirs(_deepeval_dir, exist_ok=True)

# Change cwd to the temp dir so DeepEval's relative `.deepeval` writes succeed.
# Only do this if the current directory is read-only (i.e., Lambda).
try:
    _test_path = os.path.join(os.getcwd(), ".deepeval")
    os.makedirs(_test_path, exist_ok=True)
except OSError:
    os.chdir(_tmp_dir)

# Increase recursion limit to handle nest_asyncio + Pydantic v2 + DeepEval async
# in Jupyter environments. The default 1000 is too low for the nested call stack.
if sys.getrecursionlimit() < 5000:
    sys.setrecursionlimit(5000)

logger = get_logger(__name__)


def _messages_to_deepeval_turns(messages: list) -> list:
    """Convert UAEF messages to DeepEval `Turn` objects.

    Walks `messages` in order. USER → Turn(role="user", content=...);
    ASSISTANT → Turn(role="assistant", ...); SYSTEM and TOOL messages are
    skipped — DeepEval's conversational shape doesn't model them, and the
    persona is supplied via `ConversationalTestCase.chatbot_role`, not as
    a turn.

    Imports DeepEval lazily so the module remains importable when
    DeepEval isn't installed.
    """
    from deepeval.test_case import Turn
    from uaef.models.message import MessageRole

    turns: list = []
    for msg in messages:
        if msg.role == MessageRole.USER:
            turns.append(Turn(role="user", content=msg.content))
        elif msg.role == MessageRole.ASSISTANT:
            turns.append(Turn(role="assistant", content=msg.content))
        # SYSTEM / TOOL: skip
    return turns


class DeepEvalConnector:
    """
    Connector for DeepEval library.
    
    This connector wraps DeepEval library metrics in UAEF's interface, transforming
    UAEF data to DeepEval format, calling DeepEval metrics, and transforming results
    back to UAEF MetricScore objects.
    
    DeepEval is an external evaluation library that provides metrics for contextual
    precision/recall, hallucination detection, tool correctness, and G-Eval (custom
    criteria evaluation).
    """
    
    def __init__(self, model=None):
        """Initialize DeepEval connector.
        
        Args:
            model: DeepEval model to use as judge. If None, auto-creates an
                   AmazonBedrockModel instance using UAEF config.
        """
        self._deepeval_available = False
        self._model = model
        try:
            import deepeval  # noqa: F401
            self._deepeval_available = True
            logger.info("DeepEval library loaded successfully")
        except (ImportError, TypeError, Exception) as e:
            logger.warning(
                f"DeepEval library not available: {e}. "
                "Install with: pip install 'uaef[integrations]'"
            )

        # Auto-configure Bedrock model if not provided
        if self._deepeval_available and self._model is None:
            self._auto_configure_bedrock()

    def _ensure_available(self) -> None:
        """Raise an actionable ImportError when the DeepEval backend is missing.

        Requesting a DeepEval metric without the 'integrations' extra raises an
        ImportError naming the extra and the exact ``pip install`` command,
        and returns no metric result (Requirement 3.5).
        """
        if not self._deepeval_available:
            raise integration_missing_error(_DEEPEVAL_FEATURE)

    def _auto_configure_bedrock(self) -> None:
        """Auto-configure Amazon Bedrock model from UAEF config for DeepEval."""
        try:
            from uaef.config import get_config
            from deepeval.models import AmazonBedrockModel

            config = get_config()
            self._model = AmazonBedrockModel(
                model=config.llm_judge.model_id,
                region=config.aws.region,
            )
            logger.info(f"Auto-configured DeepEval Bedrock model: {config.llm_judge.model_id}")
        except Exception as e:
            logger.warning(
                f"Could not auto-configure Bedrock for DeepEval: {e}. "
                "DeepEval will fall back to OpenAI. "
                "Set OPENAI_API_KEY or pass a model to DeepEvalConnector()."
            )
    
    def is_available(self) -> bool:
        """
        Check if DeepEval library is available.
        
        Returns:
            True if DeepEval is installed and available, False otherwise
        """
        return self._deepeval_available
    def list_available_metrics(self) -> List[str]:
        """
        List all metrics supported by this connector.

        Returns:
            List of metric names
        """
        return [
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
            "hallucination",
            "faithfulness",
            "answer_relevancy",
            "tool_correctness",
            "geval",
        ]

    
    def transform_to_deepeval_format(
        self, evaluation_input: EvaluationInput
    ) -> Dict[str, Any]:
        """
        Transform UAEF EvaluationInput to DeepEval format.
        
        DeepEval expects data in a specific format with fields like:
        - input: The user's question/query
        - actual_output: The agent's response
        - retrieval_context: List of context documents
        - expected_output: Expected correct answer
        - context: List of context strings for hallucination detection
        - tools_called: List of tool calls made by the agent
        - expected_tools: List of expected tool calls
        
        Args:
            evaluation_input: UAEF evaluation input
            
        Returns:
            Dictionary in DeepEval format
            
        Raises:
            ValueError: If required fields are missing
        """
        self._ensure_available()
        
        trace = evaluation_input.trace
        ground_truth = evaluation_input.ground_truth
        
        # Extract question (first user message)
        question = None
        actual_output = None
        
        for message in trace.messages:
            if message.role == "user" and question is None:
                question = message.content
            elif message.role == "assistant" and actual_output is None:
                actual_output = message.content
        
        # Fallback: use first message content as question if no user role found
        if not question and trace.messages:
            question = trace.messages[0].content
            logger.warning("No user message found in trace, using first message as question")
        
        if not question:
            raise ValueError("No messages found in trace")
        
        # Fallback: use last message content as answer if no assistant role found
        if not actual_output and len(trace.messages) > 1:
            actual_output = trace.messages[-1].content
            logger.warning("No assistant message found in trace, using last message as answer")
        
        if not actual_output:
            raise ValueError("No assistant response found in trace")
        
        # Build DeepEval format
        deepeval_data = {
            "input": question,
            "actual_output": actual_output,
            "retrieval_context": evaluation_input.context,
            "context": evaluation_input.context,  # For hallucination detection
        }
        
        # Add ground truth if available
        if ground_truth:
            if ground_truth.expected_output:
                deepeval_data["expected_output"] = ground_truth.expected_output
            
            if ground_truth.expected_tool_calls:
                deepeval_data["expected_tools"] = [
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                    for tc in ground_truth.expected_tool_calls
                ]
        
        # Add actual tool calls
        if trace.tool_calls:
            deepeval_data["tools_called"] = [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in trace.tool_calls
            ]
        
        return deepeval_data
    
    def calculate_contextual_precision(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate contextual precision using DeepEval.
        
        Contextual precision measures whether the retrieved context is relevant
        to the query and whether irrelevant context is ranked lower.
        
        Args:
            evaluation_input: UAEF evaluation input
            
        Returns:
            MetricScore with contextual precision score
            
        Raises:
            ImportError: If the DeepEval backend ('integrations' extra) is not installed
            ValueError: If required fields are missing
        """
        self._ensure_available()
        
        if not evaluation_input.context:
            return MetricScore(
                metric_name="deepeval_contextual_precision",
                score=None,
                reasoning="Not applicable: no retrieval context provided (metric requires RAG context)",
                metadata={"warning": "not_applicable", "reason": "no_context"},
            )

        try:
            from deepeval.metrics import ContextualPrecisionMetric
            from deepeval.test_case import LLMTestCase
            
            # Transform to DeepEval format
            deepeval_data = self.transform_to_deepeval_format(evaluation_input)
            
            if not deepeval_data.get("expected_output"):
                raise ValueError("Ground truth expected_output is required for contextual precision")
            
            # Create test case for DeepEval
            test_case = LLMTestCase(
                input=deepeval_data["input"],
                actual_output=deepeval_data["actual_output"],
                expected_output=deepeval_data["expected_output"],
                retrieval_context=deepeval_data["retrieval_context"],
            )
            
            # Calculate metric
            metric = ContextualPrecisionMetric(model=self._model)
            metric.measure(test_case)
            
            return MetricScore(
                metric_name="deepeval_contextual_precision",
                score=float(metric.score),
                reasoning=f"DeepEval contextual precision: {metric.reason if hasattr(metric, 'reason') else 'measures relevance of retrieved context'}",
                metadata={
                    "library": "deepeval",
                    "metric": "contextual_precision",
                    "contexts_count": len(deepeval_data["retrieval_context"]),
                },
            )
            
        except Exception as e:
            logger.error(f"Error calculating DeepEval contextual precision: {e}")
            raise
    
    def calculate_contextual_recall(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate contextual recall using DeepEval.
        
        Contextual recall measures whether all relevant information from the
        expected output can be found in the retrieved context.
        
        Args:
            evaluation_input: UAEF evaluation input
            
        Returns:
            MetricScore with contextual recall score
            
        Raises:
            ImportError: If the DeepEval backend ('integrations' extra) is not installed
            ValueError: If ground truth is not provided
        """
        self._ensure_available()
        
        if not evaluation_input.context:
            return MetricScore(
                metric_name="deepeval_contextual_recall",
                score=None,
                reasoning="Not applicable: no retrieval context provided (metric requires RAG context)",
                metadata={"warning": "not_applicable", "reason": "no_context"},
            )

        if not evaluation_input.ground_truth or not evaluation_input.ground_truth.expected_output:
            raise ValueError("Ground truth expected_output is required for contextual recall")
        
        try:
            from deepeval.metrics import ContextualRecallMetric
            from deepeval.test_case import LLMTestCase
            
            # Transform to DeepEval format
            deepeval_data = self.transform_to_deepeval_format(evaluation_input)
            
            # Create test case for DeepEval
            test_case = LLMTestCase(
                input=deepeval_data["input"],
                actual_output=deepeval_data["actual_output"],
                expected_output=deepeval_data["expected_output"],
                retrieval_context=deepeval_data["retrieval_context"],
            )
            
            # Calculate metric
            metric = ContextualRecallMetric(model=self._model)
            metric.measure(test_case)
            
            return MetricScore(
                metric_name="deepeval_contextual_recall",
                score=float(metric.score),
                reasoning=f"DeepEval contextual recall: {metric.reason if hasattr(metric, 'reason') else 'measures coverage of expected output in context'}",
                metadata={
                    "library": "deepeval",
                    "metric": "contextual_recall",
                    "has_ground_truth": True,
                },
            )
            
        except Exception as e:
            logger.error(f"Error calculating DeepEval contextual recall: {e}")
            raise
    
    def calculate_hallucination(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate hallucination score using DeepEval.
        
        Hallucination metric detects claims in the agent's response that are not
        supported by the provided context documents.
        
        Args:
            evaluation_input: UAEF evaluation input
            
        Returns:
            MetricScore with hallucination score (1.0 = no hallucination, 0.0 = high hallucination)
            
        Raises:
            ImportError: If the DeepEval backend ('integrations' extra) is not installed
            ValueError: If context is not provided
        """
        self._ensure_available()
        
        if not evaluation_input.context:
            return MetricScore(
                metric_name="deepeval_hallucination",
                score=None,
                reasoning="Not applicable: no retrieval context provided (metric requires RAG context)",
                metadata={"warning": "not_applicable", "reason": "no_context"},
            )
        
        try:
            from deepeval.metrics import HallucinationMetric
            from deepeval.test_case import LLMTestCase
            
            # Transform to DeepEval format
            deepeval_data = self.transform_to_deepeval_format(evaluation_input)
            
            # Create test case for DeepEval
            test_case = LLMTestCase(
                input=deepeval_data["input"],
                actual_output=deepeval_data["actual_output"],
                context=deepeval_data["context"],
            )
            
            # Calculate metric
            metric = HallucinationMetric(model=self._model)
            metric.measure(test_case)
            
            return MetricScore(
                metric_name="deepeval_hallucination",
                score=float(metric.score),
                reasoning=f"DeepEval hallucination: {metric.reason if hasattr(metric, 'reason') else 'detects unsupported claims in response'}",
                metadata={
                    "library": "deepeval",
                    "metric": "hallucination",
                    "contexts_count": len(deepeval_data["context"]),
                },
            )
            
        except Exception as e:
            logger.error(f"Error calculating DeepEval hallucination: {e}")
            raise
    
    def calculate_tool_correctness(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate tool correctness using DeepEval.
        
        Tool correctness measures how accurately the agent selected and used
        tools compared to the expected tool calls.
        
        Args:
            evaluation_input: UAEF evaluation input
            
        Returns:
            MetricScore with tool correctness score
            
        Raises:
            ImportError: If the DeepEval backend ('integrations' extra) is not installed
            ValueError: If ground truth tool calls are not provided
        """
        self._ensure_available()
        
        if not evaluation_input.ground_truth or not evaluation_input.ground_truth.expected_tool_calls:
            raise ValueError("Ground truth expected_tool_calls is required for tool correctness")
        
        try:
            from deepeval.metrics import ToolCorrectnessMetric
            from deepeval.test_case import LLMTestCase, ToolCall as DeepEvalToolCall
            
            # Transform to DeepEval format
            deepeval_data = self.transform_to_deepeval_format(evaluation_input)
            
            if "tools_called" not in deepeval_data or "expected_tools" not in deepeval_data:
                # Agent made no tool calls but ground truth expected some → score is 0
                actual = deepeval_data.get("tools_called", [])
                expected = deepeval_data.get("expected_tools", [])
                if not actual and expected:
                    return MetricScore(
                        metric_name="deepeval_tool_correctness",
                        score=0.0,
                        reasoning="Agent made no tool calls but ground truth expected tool usage",
                        metadata={
                            "library": "deepeval",
                            "metric": "tool_correctness",
                            "actual_tool_calls": 0,
                            "expected_tool_calls": len(expected),
                        },
                    )
                raise ValueError("Tool calls and expected tools are required")
            
            # Convert to DeepEval ToolCall objects
            tools_called = [
                DeepEvalToolCall(
                    name=tc["name"],
                    input_parameters=tc.get("arguments", {}),
                )
                for tc in deepeval_data["tools_called"]
            ]
            expected_tools = [
                DeepEvalToolCall(
                    name=tc["name"],
                    input_parameters=tc.get("arguments", {}),
                )
                for tc in deepeval_data["expected_tools"]
            ]

            # Create test case for DeepEval
            test_case = LLMTestCase(
                input=deepeval_data["input"],
                actual_output=deepeval_data["actual_output"],
                tools_called=tools_called,
                expected_tools=expected_tools,
            )
            
            # Calculate metric
            metric = ToolCorrectnessMetric(model=self._model)
            metric.measure(test_case)
            
            return MetricScore(
                metric_name="deepeval_tool_correctness",
                score=float(metric.score),
                reasoning=f"DeepEval tool correctness: {metric.reason if hasattr(metric, 'reason') else 'measures correctness of tool selection and usage'}",
                metadata={
                    "library": "deepeval",
                    "metric": "tool_correctness",
                    "actual_tool_calls": len(deepeval_data["tools_called"]),
                    "expected_tool_calls": len(deepeval_data["expected_tools"]),
                },
            )
            
        except Exception as e:
            logger.error(f"Error calculating DeepEval tool correctness: {e}")
            raise
    
    def calculate_contextual_relevancy(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate contextual relevancy using DeepEval.
        
        Contextual relevancy assesses whether the retrieved context is relevant
        to the user's input query.
        
        Args:
            evaluation_input: UAEF evaluation input (requires context)
            
        Returns:
            MetricScore with contextual relevancy score
        """
        self._ensure_available()

        if not evaluation_input.context:
            return MetricScore(
                metric_name="deepeval_contextual_relevancy",
                score=None,
                reasoning="Not applicable: no retrieval context provided (metric requires RAG context)",
                metadata={"warning": "not_applicable", "reason": "no_context"},
            )

        try:
            from deepeval.metrics import ContextualRelevancyMetric
            from deepeval.test_case import LLMTestCase

            deepeval_data = self.transform_to_deepeval_format(evaluation_input)

            test_case = LLMTestCase(
                input=deepeval_data["input"],
                actual_output=deepeval_data["actual_output"],
                retrieval_context=deepeval_data["retrieval_context"],
            )

            metric = ContextualRelevancyMetric(model=self._model)
            metric.measure(test_case)

            return MetricScore(
                metric_name="deepeval_contextual_relevancy",
                score=float(metric.score),
                reasoning=f"DeepEval contextual relevancy: {getattr(metric, 'reason', 'assesses relevance of retrieved context to query')}",
                metadata={
                    "library": "deepeval",
                    "metric": "contextual_relevancy",
                    "contexts_count": len(deepeval_data["retrieval_context"]),
                },
            )

        except Exception as e:
            logger.error(f"Error calculating DeepEval contextual relevancy: {e}")
            raise

    def calculate_faithfulness(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate faithfulness using DeepEval.
        
        Faithfulness measures whether the generated answer is factually consistent
        with the provided retrieval context (i.e., no hallucinated claims).
        
        Args:
            evaluation_input: UAEF evaluation input (requires context)
            
        Returns:
            MetricScore with faithfulness score
        """
        self._ensure_available()

        if not evaluation_input.context:
            return MetricScore(
                metric_name="deepeval_faithfulness",
                score=None,
                reasoning="Not applicable: no retrieval context provided (metric requires RAG context)",
                metadata={"warning": "not_applicable", "reason": "no_context"},
            )

        try:
            from deepeval.metrics import FaithfulnessMetric
            from deepeval.test_case import LLMTestCase

            deepeval_data = self.transform_to_deepeval_format(evaluation_input)

            test_case = LLMTestCase(
                input=deepeval_data["input"],
                actual_output=deepeval_data["actual_output"],
                retrieval_context=deepeval_data["retrieval_context"],
            )

            metric = FaithfulnessMetric(model=self._model)
            metric.measure(test_case)

            return MetricScore(
                metric_name="deepeval_faithfulness",
                score=float(metric.score),
                reasoning=f"DeepEval faithfulness: {getattr(metric, 'reason', 'measures factual consistency with retrieval context')}",
                metadata={
                    "library": "deepeval",
                    "metric": "faithfulness",
                    "contexts_count": len(deepeval_data["retrieval_context"]),
                },
            )

        except Exception as e:
            logger.error(f"Error calculating DeepEval faithfulness: {e}")
            raise

    def calculate_answer_relevancy(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate answer relevancy using DeepEval.
        
        Answer relevancy evaluates whether the generated answer is relevant
        to the user's input query.
        
        Args:
            evaluation_input: UAEF evaluation input
            
        Returns:
            MetricScore with answer relevancy score
        """
        self._ensure_available()

        try:
            from deepeval.metrics import AnswerRelevancyMetric
            from deepeval.test_case import LLMTestCase

            deepeval_data = self.transform_to_deepeval_format(evaluation_input)

            test_case = LLMTestCase(
                input=deepeval_data["input"],
                actual_output=deepeval_data["actual_output"],
            )

            metric = AnswerRelevancyMetric(model=self._model)
            metric.measure(test_case)

            return MetricScore(
                metric_name="deepeval_answer_relevancy",
                score=float(metric.score),
                reasoning=f"DeepEval answer relevancy: {getattr(metric, 'reason', 'evaluates relevance of answer to query')}",
                metadata={
                    "library": "deepeval",
                    "metric": "answer_relevancy",
                },
            )

        except Exception as e:
            logger.error(f"Error calculating DeepEval answer relevancy: {e}")
            raise

    def calculate_geval(
        self,
        evaluation_input: EvaluationInput,
        criteria: str,
        evaluation_steps: Optional[List[str]] = None,
    ) -> MetricScore:
        """
        Calculate G-Eval score using DeepEval with custom criteria.
        
        G-Eval is a framework for LLM-based evaluation using custom criteria.
        It allows defining specific evaluation criteria and steps.
        
        Args:
            evaluation_input: UAEF evaluation input
            criteria: Custom evaluation criteria description
            evaluation_steps: Optional list of evaluation steps to follow
            
        Returns:
            MetricScore with G-Eval score
            
        Raises:
            ImportError: If the DeepEval backend ('integrations' extra) is not installed
            ValueError: If required fields are missing
        """
        self._ensure_available()
        
        if not criteria:
            raise ValueError("Criteria is required for G-Eval")
        
        try:
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCase
            
            # Transform to DeepEval format
            deepeval_data = self.transform_to_deepeval_format(evaluation_input)
            
            # Create test case for DeepEval
            test_case = LLMTestCase(
                input=deepeval_data["input"],
                actual_output=deepeval_data["actual_output"],
            )
            
            # Add expected output if available
            if deepeval_data.get("expected_output"):
                test_case.expected_output = deepeval_data["expected_output"]
            
            # Create G-Eval metric with custom criteria
            metric_kwargs = {
                "name": "custom_geval",
                "criteria": criteria,
                "evaluation_params": [
                    LLMTestCase.input,
                    LLMTestCase.actual_output,
                ],
            }
            
            if evaluation_steps:
                metric_kwargs["evaluation_steps"] = evaluation_steps
            
            metric = GEval(**metric_kwargs)
            metric.measure(test_case)
            
            return MetricScore(
                metric_name="deepeval_geval",
                score=float(metric.score),
                reasoning=f"DeepEval G-Eval: {metric.reason if hasattr(metric, 'reason') else criteria}",
                metadata={
                    "library": "deepeval",
                    "metric": "geval",
                    "criteria": criteria,
                    "has_evaluation_steps": evaluation_steps is not None,
                },
            )
            
        except Exception as e:
            logger.error(f"Error calculating DeepEval G-Eval: {e}")
            raise
    
    def calculate_role_adherence(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """Calculate role adherence using DeepEval's `RoleAdherenceMetric`.

        Reads the chatbot persona strictly from
        `ground_truth.expected_arguments["chatbot_role"]`. Builds a
        `ConversationalTestCase` from `trace.messages`, delegates to
        DeepEval, and surfaces the score along with per-turn out-of-character
        verdicts in metadata.

        Returns a `not_applicable` `MetricScore` if `chatbot_role` is missing
        or the filtered turn list is empty. Errors raised by DeepEval's
        `measure()` propagate to the caller (matches sibling methods).
        """
        if not self._deepeval_available:
            raise RuntimeError(
                "DeepEval library not available. Install with: pip install deepeval"
            )

        gt = evaluation_input.ground_truth
        role = (
            gt.expected_arguments.get("chatbot_role")
            if gt is not None and gt.expected_arguments
            else None
        )
        if not role:
            return MetricScore(
                metric_name="deepeval_role_adherence",
                score=None,
                reasoning="Cannot evaluate: expected_arguments['chatbot_role'] is required",
                metadata={"warning": "not_applicable"},
            )

        turns = _messages_to_deepeval_turns(evaluation_input.trace.messages)
        if not turns:
            return MetricScore(
                metric_name="deepeval_role_adherence",
                score=None,
                reasoning="Cannot evaluate: no user/assistant messages",
                metadata={"warning": "not_applicable"},
            )

        from deepeval.metrics import RoleAdherenceMetric
        from deepeval.test_case import ConversationalTestCase

        test_case = ConversationalTestCase(turns=turns, chatbot_role=role)
        metric = RoleAdherenceMetric(model=self._model)
        try:
            metric.measure(test_case)
        except Exception as e:
            logger.error(f"Error calculating DeepEval role adherence: {e}")
            raise

        # Per-verdict access uses getattr for resilience against DeepEval field
        # renames; outer attributes are required and we want loud failures.
        verdicts = metric.out_of_character_verdicts.verdicts or []
        out_of_character_turns = [
            {
                "turn_index": getattr(v, "index", None),
                "ai_message": getattr(v, "ai_message", None),
                "reason": getattr(v, "reason", None),
            }
            for v in verdicts
        ]

        assistant_turn_count = sum(1 for t in turns if t.role == "assistant")

        if metric.score is None:
            return MetricScore(
                metric_name="deepeval_role_adherence",
                score=None,
                reasoning="Cannot evaluate: DeepEval returned no score (likely zero assistant turns)",
                metadata={
                    "warning": "not_applicable",
                    "library": "deepeval",
                    "metric": "role_adherence",
                    "out_of_character_turns": out_of_character_turns,
                    "assistant_turn_count": assistant_turn_count,
                },
            )

        return MetricScore(
            metric_name="deepeval_role_adherence",
            score=float(metric.score),
            reasoning=(
                getattr(metric, "reason", None)
                or "DeepEval role adherence (no reason returned)"
            ),
            metadata={
                "library": "deepeval",
                "metric": "role_adherence",
                "out_of_character_turns": out_of_character_turns,
                "assistant_turn_count": assistant_turn_count,
            },
        )

    def calculate_metrics(
        self,
        evaluation_input: EvaluationInput,
        metric_names: List[str],
        geval_criteria: Optional[str] = None,
    ) -> List[MetricScore]:
        """
        Calculate multiple DeepEval metrics at once.
        
        Args:
            evaluation_input: UAEF evaluation input
            metric_names: List of metric names to calculate
                         (e.g., ["contextual_precision", "hallucination"])
            geval_criteria: Optional criteria for G-Eval metric
            
        Returns:
            List of MetricScore objects
            
        Raises:
            ImportError: If the DeepEval backend ('integrations' extra) is not installed
            ValueError: If an unknown metric name is provided
        """
        self._ensure_available()
        
        metric_map = {
            "contextual_precision": self.calculate_contextual_precision,
            "contextual_recall": self.calculate_contextual_recall,
            "contextual_relevancy": self.calculate_contextual_relevancy,
            "hallucination": self.calculate_hallucination,
            "faithfulness": self.calculate_faithfulness,
            "answer_relevancy": self.calculate_answer_relevancy,
            "tool_correctness": self.calculate_tool_correctness,
        }
        
        scores = []
        for metric_name in metric_names:
            if metric_name == "geval":
                if not geval_criteria:
                    logger.warning("G-Eval requires criteria, skipping")
                    continue
                try:
                    score = self.calculate_geval(evaluation_input, geval_criteria)
                    scores.append(score)
                except Exception as e:
                    logger.error(f"Error calculating geval: {e}")
                    # Continue with remaining metrics per requirement 14.6
                    continue
            elif metric_name in metric_map:
                try:
                    score = metric_map[metric_name](evaluation_input)
                    scores.append(score)
                except Exception as e:
                    logger.error(f"Error calculating {metric_name}: {e}")
                    # Continue with remaining metrics per requirement 14.6
                    continue
            else:
                logger.warning(f"Unknown DeepEval metric: {metric_name}, skipping")
                continue
        
        return scores
