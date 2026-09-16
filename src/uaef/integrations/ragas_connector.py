# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""RAGAS connector for external metric integration."""

import sys
from typing import Any, Dict, List, Optional

from uaef.models.evaluation_input import EvaluationInput
from uaef.models.metric_score import MetricScore
from uaef.logging import get_logger
from uaef.integrations._guards import integration_missing_error

# Human-readable label used in the actionable ImportError raised when the
# RAGAS backend (shipped behind the 'integrations' extra) is not installed.
_RAGAS_FEATURE = "RAGAS metrics"

# Increase recursion limit to handle nest_asyncio + Pydantic v2 + RAGAS async executor
# in Jupyter environments. The default 1000 is too low for the nested call stack.
if sys.getrecursionlimit() < 5000:
    sys.setrecursionlimit(5000)

logger = get_logger(__name__)


class RAGASConnector:
    """
    Connector for RAGAS (Retrieval Augmented Generation Assessment) library.
    
    This connector wraps RAGAS library metrics in UAEF's interface, transforming
    UAEF data to RAGAS format, calling RAGAS metrics, and transforming results
    back to UAEF MetricScore objects.
    
    RAGAS is an external evaluation library for RAG systems that provides metrics
    for answer quality and tool calling accuracy.
    """
    
    def __init__(self, llm=None, embeddings=None):
        """Initialize RAGAS connector.
        
        Args:
            llm: LangChain LLM to use as judge. If None, auto-creates a ChatBedrock
                 instance using UAEF config and environment AWS credentials.
            embeddings: LangChain embeddings model. If None, auto-creates a
                        BedrockEmbeddings instance using UAEF config.
        """
        self._ragas_available = False
        self._llm = llm
        self._embeddings = embeddings
        try:
            # The RAGAS metric path needs both ``ragas`` and ``datasets``
            # (both provided by the 'integrations' extra), so treat the
            # backend as available only when the full stack imports.
            import ragas  # noqa: F401
            import datasets  # noqa: F401
            self._ragas_available = True
            logger.info("RAGAS library loaded successfully")
        except ImportError:
            logger.warning(
                "RAGAS library not available. "
                "Install with: pip install 'uaef[integrations]'"
            )

        # Auto-configure Bedrock LLM/embeddings if not provided
        if self._ragas_available and (self._llm is None or self._embeddings is None):
            self._auto_configure_bedrock()

    def _ensure_available(self) -> None:
        """Raise an actionable ImportError when the RAGAS backend is missing.

        Requesting a RAGAS metric without the 'integrations' extra raises an
        ImportError naming the extra and the exact ``pip install`` command,
        and returns no metric result (Requirement 3.5).
        """
        if not self._ragas_available:
            raise integration_missing_error(_RAGAS_FEATURE)
    
    def _auto_configure_bedrock(self) -> None:
        """Auto-configure Bedrock LLM and embeddings from UAEF config."""
        try:
            from uaef.config import get_config
            import boto3
            from botocore.config import Config as BotoConfig

            config = get_config()
            boto_config = BotoConfig(
                region_name=config.aws.region,
                retries={"max_attempts": 3, "mode": "standard"},
            )
            bedrock_client = boto3.client(service_name="bedrock-runtime", config=boto_config)

            if self._llm is None:
                from langchain_aws import ChatBedrock
                self._llm = ChatBedrock(
                    client=bedrock_client,
                    model_id=config.llm_judge.model_id,
                    model_kwargs={
                        "max_tokens": config.llm_judge.max_tokens,
                        "temperature": config.llm_judge.temperature,
                    },
                )
                logger.info(f"Auto-configured RAGAS LLM judge: {config.llm_judge.model_id}")

            if self._embeddings is None:
                from langchain_aws import BedrockEmbeddings
                self._embeddings = BedrockEmbeddings(
                    client=bedrock_client,
                    model_id="amazon.titan-embed-text-v2:0",
                )
                logger.info("Auto-configured RAGAS embeddings: amazon.titan-embed-text-v2:0")

        except Exception as e:
            logger.warning(
                f"Could not auto-configure Bedrock for RAGAS: {e}. "
                "RAGAS will fall back to its default LLM (OpenAI). "
                "Set OPENAI_API_KEY or pass llm/embeddings to RAGASConnector()."
            )

    def is_available(self) -> bool:
        """
        Check if RAGAS library is available.
        
        Returns:
            True if RAGAS is installed and available, False otherwise
        """
        return self._ragas_available

    def _evaluate_kwargs(self) -> Dict[str, Any]:
        """Build extra kwargs for ragas evaluate() with optional llm/embeddings."""
        kwargs: Dict[str, Any] = {}
        if self._llm is not None:
            kwargs["llm"] = self._llm
        if self._embeddings is not None:
            kwargs["embeddings"] = self._embeddings
        return kwargs

    @staticmethod
    def _extract_score(result: Any, metric_key: str) -> float:
        """Extract a single float score from a RAGAS evaluate result.
        
        RAGAS may return a single float or a list of floats depending on version.
        Returns nan if the score cannot be extracted or is nan.
        """
        value = result[metric_key]
        if isinstance(value, list):
            value = value[0] if value else 0.0
        return float(value)
    def list_available_metrics(self) -> List[str]:
        """
        List all metrics supported by this connector.

        Returns:
            List of metric names
        """
        return [
            "faithfulness",
            "context_precision",
            "context_recall",
            "answer_precision",
            "answer_recall",
            "answer_correctness",
            "tool_call_accuracy",
        ]

    
    def transform_to_ragas_format(
        self, evaluation_input: EvaluationInput
    ) -> Dict[str, Any]:
        """
        Transform UAEF EvaluationInput to RAGAS format.
        
        RAGAS expects data in a specific format with fields like:
        - question: The user's question/query
        - answer: The agent's response
        - contexts: List of context documents
        - ground_truth: Expected correct answer
        - reference: Expected tool calls and arguments
        
        Args:
            evaluation_input: UAEF evaluation input
            
        Returns:
            Dictionary in RAGAS format
            
        Raises:
            ValueError: If required fields are missing
        """
        self._ensure_available()
        
        trace = evaluation_input.trace
        ground_truth = evaluation_input.ground_truth
        
        # Extract question (first user message)
        question = None
        answer = None
        
        for message in trace.messages:
            if message.role == "user" and question is None:
                question = message.content
            elif message.role == "assistant" and answer is None:
                answer = message.content
        
        # Fallback: use first message content as question if no user role found
        if not question and trace.messages:
            question = trace.messages[0].content
            logger.warning("No user message found in trace, using first message as question")
        
        if not question:
            raise ValueError("No messages found in trace")
        
        # Fallback: use last message content as answer if no assistant role found
        if not answer and len(trace.messages) > 1:
            answer = trace.messages[-1].content
            logger.warning("No assistant message found in trace, using last message as answer")
        
        if not answer:
            raise ValueError("No assistant response found in trace")
        
        # Build RAGAS format
        ragas_data = {
            "question": question,
            "answer": answer,
            "contexts": evaluation_input.context,
        }
        
        # Add ground truth if available
        if ground_truth:
            if ground_truth.expected_output:
                ragas_data["ground_truth"] = ground_truth.expected_output
            
            if ground_truth.expected_tool_calls:
                ragas_data["reference_tool_calls"] = [
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                    for tc in ground_truth.expected_tool_calls
                ]
        
        # Add actual tool calls
        if trace.tool_calls:
            ragas_data["tool_calls"] = [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in trace.tool_calls
            ]
        
        return ragas_data
    
    def calculate_faithfulness(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate faithfulness using RAGAS.
        
        Faithfulness measures whether every claim in the generated answer
        is supported by the retrieved context. A score of 1.0 means no
        hallucinations relative to the provided context.
        
        Args:
            evaluation_input: UAEF evaluation input (requires context)
            
        Returns:
            MetricScore with faithfulness score
        """
        self._ensure_available()

        if not evaluation_input.context:
            raise ValueError("Context is required for faithfulness")

        try:
            from ragas.metrics import faithfulness
            from ragas import evaluate
            from datasets import Dataset

            ragas_data = self.transform_to_ragas_format(evaluation_input)

            dataset = Dataset.from_dict({
                "question": [ragas_data["question"]],
                "answer": [ragas_data["answer"]],
                "contexts": [ragas_data["contexts"]],
            })

            result = evaluate(dataset, metrics=[faithfulness], **self._evaluate_kwargs())
            score = self._extract_score(result, "faithfulness")

            return MetricScore(
                metric_name="ragas_faithfulness",
                score=float(score),
                reasoning="RAGAS faithfulness: measures if answer claims are supported by context",
                metadata={
                    "library": "ragas",
                    "metric": "faithfulness",
                    "contexts_count": len(ragas_data["contexts"]),
                },
            )

        except Exception as e:
            logger.error(f"Error calculating RAGAS faithfulness: {e}")
            raise

    def calculate_context_precision(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate context precision using RAGAS.
        
        Context precision measures whether the retrieved chunks ranked highest
        are the most relevant ones. Requires ground truth to determine which
        chunks are actually useful.
        
        Args:
            evaluation_input: UAEF evaluation input (requires context and ground truth)
            
        Returns:
            MetricScore with context precision score
        """
        self._ensure_available()

        if not evaluation_input.context:
            raise ValueError("Context is required for context precision")

        if not evaluation_input.ground_truth or not evaluation_input.ground_truth.expected_output:
            raise ValueError("Ground truth expected_output is required for context precision")

        try:
            from ragas.metrics import context_precision
            from ragas import evaluate
            from datasets import Dataset

            ragas_data = self.transform_to_ragas_format(evaluation_input)

            dataset = Dataset.from_dict({
                "question": [ragas_data["question"]],
                "answer": [ragas_data["answer"]],
                "contexts": [ragas_data["contexts"]],
                "ground_truth": [ragas_data["ground_truth"]],
            })

            result = evaluate(dataset, metrics=[context_precision], **self._evaluate_kwargs())
            score = self._extract_score(result, "context_precision")

            return MetricScore(
                metric_name="ragas_context_precision",
                score=float(score),
                reasoning="RAGAS context precision: measures if top-ranked retrieved chunks are the most relevant",
                metadata={
                    "library": "ragas",
                    "metric": "context_precision",
                    "contexts_count": len(ragas_data["contexts"]),
                },
            )

        except Exception as e:
            logger.error(f"Error calculating RAGAS context precision: {e}")
            raise

    def calculate_context_recall(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate context recall using RAGAS.
        
        Context recall measures what percentage of the ground truth answer's
        claims are covered by the retrieved context. Low recall means the
        retriever is missing information the LLM needs.
        
        Args:
            evaluation_input: UAEF evaluation input (requires context and ground truth)
            
        Returns:
            MetricScore with context recall score
        """
        self._ensure_available()

        if not evaluation_input.context:
            raise ValueError("Context is required for context recall")

        if not evaluation_input.ground_truth or not evaluation_input.ground_truth.expected_output:
            raise ValueError("Ground truth expected_output is required for context recall")

        try:
            from ragas.metrics import context_recall
            from ragas import evaluate
            from datasets import Dataset

            ragas_data = self.transform_to_ragas_format(evaluation_input)

            dataset = Dataset.from_dict({
                "question": [ragas_data["question"]],
                "answer": [ragas_data["answer"]],
                "contexts": [ragas_data["contexts"]],
                "ground_truth": [ragas_data["ground_truth"]],
            })

            result = evaluate(dataset, metrics=[context_recall], **self._evaluate_kwargs())
            score = self._extract_score(result, "context_recall")

            return MetricScore(
                metric_name="ragas_context_recall",
                score=float(score),
                reasoning="RAGAS context recall: measures coverage of ground truth claims in retrieved context",
                metadata={
                    "library": "ragas",
                    "metric": "context_recall",
                    "contexts_count": len(ragas_data["contexts"]),
                },
            )

        except Exception as e:
            logger.error(f"Error calculating RAGAS context recall: {e}")
            raise

    def calculate_answer_precision(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate answer precision using RAGAS.
        
        Answer precision measures how much of the generated answer is relevant
        and supported by the context documents.
        
        Args:
            evaluation_input: UAEF evaluation input
            
        Returns:
            MetricScore with precision score
            
        Raises:
            ImportError: If the RAGAS backend ('integrations' extra) is not installed
            ValueError: If required fields are missing
        """
        self._ensure_available()
        
        try:
            from ragas.metrics import answer_relevancy
            from ragas import evaluate
            from datasets import Dataset
            
            # Transform to RAGAS format
            ragas_data = self.transform_to_ragas_format(evaluation_input)
            
            # Create dataset for RAGAS
            dataset = Dataset.from_dict({
                "question": [ragas_data["question"]],
                "answer": [ragas_data["answer"]],
                "contexts": [ragas_data["contexts"]],
            })
            
            # Calculate metric
            result = evaluate(dataset, metrics=[answer_relevancy], **self._evaluate_kwargs())
            score = self._extract_score(result, "answer_relevancy")
            
            import math
            if math.isnan(score):
                score = 0.0
            
            return MetricScore(
                metric_name="ragas_answer_precision",
                score=float(score),
                reasoning="RAGAS answer precision: measures relevance and support from context",
                metadata={
                    "library": "ragas",
                    "metric": "answer_relevancy",
                    "contexts_count": len(ragas_data["contexts"]),
                },
            )
            
        except Exception as e:
            logger.error(f"Error calculating RAGAS answer precision: {e}")
            raise
    
    def calculate_answer_recall(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate answer recall using RAGAS.
        
        Answer recall measures how much of the ground truth information is
        present in the generated answer.
        
        Args:
            evaluation_input: UAEF evaluation input
            
        Returns:
            MetricScore with recall score
            
        Raises:
            ImportError: If the RAGAS backend ('integrations' extra) is not installed
            ValueError: If ground truth is not provided
        """
        self._ensure_available()
        
        if not evaluation_input.ground_truth or not evaluation_input.ground_truth.expected_output:
            raise ValueError("Ground truth expected_output is required for answer recall")
        
        try:
            from ragas.metrics import answer_correctness
            from ragas import evaluate
            from datasets import Dataset
            
            # Transform to RAGAS format
            ragas_data = self.transform_to_ragas_format(evaluation_input)
            
            # Create dataset for RAGAS
            dataset = Dataset.from_dict({
                "question": [ragas_data["question"]],
                "answer": [ragas_data["answer"]],
                "ground_truth": [ragas_data["ground_truth"]],
            })
            
            # Calculate metric
            result = evaluate(dataset, metrics=[answer_correctness], **self._evaluate_kwargs())
            score = self._extract_score(result, "answer_correctness")
            
            import math
            if math.isnan(score):
                score = 0.0
            
            return MetricScore(
                metric_name="ragas_answer_recall",
                score=float(score),
                reasoning="RAGAS answer recall: measures coverage of ground truth information",
                metadata={
                    "library": "ragas",
                    "metric": "answer_correctness",
                    "has_ground_truth": True,
                },
            )
            
        except Exception as e:
            logger.error(f"Error calculating RAGAS answer recall: {e}")
            raise
    
    def calculate_answer_correctness(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate answer correctness using RAGAS.
        
        Answer correctness is a comprehensive metric that combines factual
        accuracy and semantic similarity with the ground truth.
        
        Args:
            evaluation_input: UAEF evaluation input
            
        Returns:
            MetricScore with correctness score
            
        Raises:
            ImportError: If the RAGAS backend ('integrations' extra) is not installed
            ValueError: If ground truth is not provided
        """
        self._ensure_available()
        
        if not evaluation_input.ground_truth or not evaluation_input.ground_truth.expected_output:
            raise ValueError("Ground truth expected_output is required for answer correctness")
        
        try:
            from ragas.metrics import answer_correctness
            from ragas import evaluate
            from datasets import Dataset
            
            # Transform to RAGAS format
            ragas_data = self.transform_to_ragas_format(evaluation_input)
            
            # Create dataset for RAGAS
            dataset = Dataset.from_dict({
                "question": [ragas_data["question"]],
                "answer": [ragas_data["answer"]],
                "ground_truth": [ragas_data["ground_truth"]],
            })
            
            # Calculate metric
            result = evaluate(dataset, metrics=[answer_correctness], **self._evaluate_kwargs())
            score = self._extract_score(result, "answer_correctness")
            
            import math
            if math.isnan(score):
                score = 0.0
            
            return MetricScore(
                metric_name="ragas_answer_correctness",
                score=float(score),
                reasoning="RAGAS answer correctness: combines factual accuracy and semantic similarity",
                metadata={
                    "library": "ragas",
                    "metric": "answer_correctness",
                    "has_ground_truth": True,
                },
            )
            
        except Exception as e:
            logger.error(f"Error calculating RAGAS answer correctness: {e}")
            raise
    
    def calculate_tool_call_accuracy(
        self, evaluation_input: EvaluationInput
    ) -> MetricScore:
        """
        Calculate tool call accuracy using RAGAS.
        
        Tool call accuracy measures how accurately the agent selected and
        used tools compared to the expected tool calls.
        
        RAGAS 0.2.x requires a MultiTurnSample for this metric, so we build
        a conversation with HumanMessage → AIMessage (with tool_calls) →
        ToolMessage sequence.
        
        Args:
            evaluation_input: UAEF evaluation input
            
        Returns:
            MetricScore with tool call accuracy score
            
        Raises:
            ImportError: If the RAGAS backend ('integrations' extra) is not installed
            ValueError: If ground truth tool calls are not provided
        """
        self._ensure_available()
        
        if not evaluation_input.ground_truth or not evaluation_input.ground_truth.expected_tool_calls:
            raise ValueError("Ground truth expected_tool_calls is required for tool call accuracy")
        
        try:
            from ragas.metrics import ToolCallAccuracy
            from ragas.dataset_schema import MultiTurnSample, EvaluationDataset
            from ragas.messages import HumanMessage, AIMessage, ToolMessage
            from ragas.messages import ToolCall as RagasToolCall
            from ragas import evaluate

            # Transform to RAGAS format
            ragas_data = self.transform_to_ragas_format(evaluation_input)
            
            if "tool_calls" not in ragas_data or "reference_tool_calls" not in ragas_data:
                # Agent made no tool calls but ground truth expected some → score is 0
                actual = ragas_data.get("tool_calls", [])
                expected = ragas_data.get("reference_tool_calls", [])
                if not actual and expected:
                    return MetricScore(
                        metric_name="ragas_tool_call_accuracy",
                        score=0.0,
                        reasoning="Agent made no tool calls but ground truth expected tool usage",
                        metadata={
                            "library": "ragas",
                            "metric": "tool_call_accuracy",
                            "actual_tool_calls": 0,
                            "expected_tool_calls": len(expected),
                        },
                    )
                raise ValueError("Reference tool calls are required for tool call accuracy")

            # Build MultiTurnSample conversation
            # 1. User asks the question
            user_msg = HumanMessage(content=ragas_data["question"])

            # 2. AI responds with tool calls
            actual_tool_calls = [
                RagasToolCall(
                    name=tc["name"],
                    args=tc.get("arguments", tc.get("args", {})),
                )
                for tc in ragas_data["tool_calls"]
            ]
            ai_msg = AIMessage(
                content=ragas_data.get("answer", ""),
                tool_calls=actual_tool_calls,
            )

            # 3. Tool responses (one per tool call)
            tool_messages = []
            for tc in ragas_data["tool_calls"]:
                result = tc.get("result", tc.get("response", ""))
                tool_messages.append(ToolMessage(content=str(result) if result else ""))

            # Build reference tool calls
            reference_tool_calls = [
                RagasToolCall(
                    name=tc["name"],
                    args=tc.get("arguments", tc.get("args", {})),
                )
                for tc in ragas_data["reference_tool_calls"]
            ]

            # Create the multi-turn sample
            conversation = [user_msg, ai_msg] + tool_messages
            sample = MultiTurnSample(
                user_input=conversation,
                reference_tool_calls=reference_tool_calls,
            )

            # Evaluate using the multi-turn API
            tool_call_accuracy = ToolCallAccuracy()
            dataset = EvaluationDataset(samples=[sample])
            result = evaluate(dataset, metrics=[tool_call_accuracy], **self._evaluate_kwargs())
            score = self._extract_score(result, "tool_call_accuracy")
            
            return MetricScore(
                metric_name="ragas_tool_call_accuracy",
                score=float(score),
                reasoning="RAGAS tool call accuracy: measures correctness of tool selection and usage",
                metadata={
                    "library": "ragas",
                    "metric": "tool_call_accuracy",
                    "actual_tool_calls": len(ragas_data["tool_calls"]),
                    "expected_tool_calls": len(ragas_data["reference_tool_calls"]),
                },
            )
            
        except Exception as e:
            logger.error(f"Error calculating RAGAS tool call accuracy: {e}")
            raise
    
    def calculate_metrics(
        self,
        evaluation_input: EvaluationInput,
        metric_names: List[str],
    ) -> List[MetricScore]:
        """
        Calculate multiple RAGAS metrics at once.
        
        Args:
            evaluation_input: UAEF evaluation input
            metric_names: List of metric names to calculate
                         (e.g., ["answer_precision", "answer_recall"])
            
        Returns:
            List of MetricScore objects
            
        Raises:
            ImportError: If the RAGAS backend ('integrations' extra) is not installed
            ValueError: If an unknown metric name is provided
        """
        self._ensure_available()
        
        metric_map = {
            "faithfulness": self.calculate_faithfulness,
            "context_precision": self.calculate_context_precision,
            "context_recall": self.calculate_context_recall,
            "answer_precision": self.calculate_answer_precision,
            "answer_recall": self.calculate_answer_recall,
            "answer_correctness": self.calculate_answer_correctness,
            "tool_call_accuracy": self.calculate_tool_call_accuracy,
        }
        
        scores = []
        for metric_name in metric_names:
            if metric_name not in metric_map:
                logger.warning(f"Unknown RAGAS metric: {metric_name}, skipping")
                continue
            
            try:
                score = metric_map[metric_name](evaluation_input)
                scores.append(score)
            except Exception as e:
                logger.error(f"Error calculating {metric_name}: {e}")
                # Continue with remaining metrics per requirement 14.6
                continue
        
        return scores
