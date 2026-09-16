# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
High-level Python API for UAEF.

This module provides simple, convenient functions for common UAEF workflows:
- Simple evaluation with automatic adapter selection
- Batch evaluation
- Experiment management (create, run, compare)
- Simulation (placeholder)
- HITL workflows (review queue, feedback, agreement)

The API wraps existing UAEF modules to provide the simplest possible interface
for common use cases, handling setup and configuration automatically.

Examples:
    Simple evaluation:
        >>> from uaef.api import evaluate
        >>> result = evaluate(
        ...     trace=langgraph_output,
        ...     ground_truth=expected_output,
        ...     adapter="langgraph"
        ... )
        >>> print(f"Score: {result.overall_score:.2f}")
    
    Experiment workflow:
        >>> from uaef.api import create_experiment, create_run, compare_runs
        >>> experiment = create_experiment(
        ...     name="GPT-4 Agent",
        ...     description="Testing tool calling"
        ... )
        >>> run = create_run(
        ...     experiment_id=experiment.experiment_id,
        ...     run_name="baseline-v1",
        ...     agent_design={"model": "gpt-4", "framework": "langgraph"}
        ... )
"""

import os
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from uaef.adapters.registry import get_adapter
from uaef.evaluation.offline import OfflineEvaluator
from uaef.evaluation.single_agent import SingleAgentEvaluator
from uaef.evaluation.multi_agent import MultiAgentEvaluator
from uaef.evaluation.multi_turn import MultiTurnEvaluator
from uaef.experiments.comparison import ComparisonEngine, ComparisonReport
from uaef.experiments.manager import ExperimentManager
from uaef.experiments.models import AgentDesign, Experiment, ExperimentRun
from uaef.experiments.regression import RegressionDetector, RegressionReport
from uaef.hitl.agreement import AgreementCalculator, AgreementMetrics
from uaef.hitl.models import ItemType, ReviewItem
from uaef.hitl.queue import ReviewQueue
from uaef.logging import get_logger
from uaef.models.agent_trace import AgentTrace
from uaef.models.evaluation_result import EvaluationResult
from uaef.models.ground_truth import GroundTruth
from uaef.models.multi_agent_trace import MultiAgentTrace
from uaef.metrics import get_full_metric_catalog
from uaef.storage.dynamodb_s3 import DynamoS3Storage, get_storage

logger = get_logger(__name__)

# Global instances for convenience
_experiment_manager = ExperimentManager()
_comparison_engine = ComparisonEngine()
_regression_detector = RegressionDetector()
_review_queue = ReviewQueue()
_agreement_calculator = AgreementCalculator()

# Lazy-initialized DynamoDB+S3 storage for persistence
_store: Optional[DynamoS3Storage] = None


def _get_store() -> DynamoS3Storage:
    """Get or initialize the DynamoDB+S3 storage instance."""
    global _store
    if _store is None:
        _store = get_storage()
        _store.create_table_if_not_exists()
        _store.create_bucket_if_not_exists()
    return _store


def _resolve_experiment(
    experiment_name: Optional[str] = None,
    experiment_objective: Optional[str] = None,
) -> tuple:
    """
    Create a new experiment for persistence.

    Every persist=True call creates a new experiment. If experiment_name
    is provided, it is used (multiple experiments can share the same name).
    Otherwise defaults to "default".

    Returns:
        (experiment_id: str, experiment_name: str)
    """
    from uuid import uuid4

    new_id = str(uuid4())
    name = experiment_name or "default"
    logger.info(f"Creating new experiment '{name}' with ID {new_id}")
    return new_id, name


#: Security review M-02: opt-in redaction of PII in the persisted copy of
#: judge reasoning text (and any other free-text fields in the serialized
#: EvaluationResult). Off by default — existing deployments/library callers
#: see no change unless this env var is explicitly set. Redaction happens on
#: a COPY of the already-serialized dicts passed to save_experiment, never on
#: the EvaluationResult objects returned to the caller of evaluate()/
#: batch_evaluate() — so even with this enabled, the in-memory return value a
#: library caller receives is unaffected, only what gets written to S3/
#: DynamoDB. Applied strictly after the judge has already scored the trace
#: (this function is only called from the persist branches below, after
#: evaluator.evaluate()/batch_evaluate already ran), so this cannot change
#: any score.
PII_REDACTION_ENV = "UAEF_REDACT_PII_ON_PERSIST"


def _redact_result_dicts_for_persistence(
    result_dicts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return a redacted copy of serialized EvaluationResult dicts, if enabled.

    No-op (returns the input as-is) unless UAEF_REDACT_PII_ON_PERSIST is set.
    Redacts every ``metric_scores[*].reasoning`` string in place on a deep
    copy — this is the only free-text field in the serialized shape that can
    contain PII quoted from an agent's response (scores/names/timestamps
    carry no PII risk). See ``uaef.security.pii.redact_for_storage``.
    """
    if os.environ.get(PII_REDACTION_ENV, "").strip().lower() not in {"true", "1", "yes"}:
        return result_dicts
    try:
        from uaef.security.pii import redact_for_storage
    except Exception:  # noqa: BLE001 — redaction must never break persistence
        return result_dicts

    import copy

    redacted = copy.deepcopy(result_dicts)
    for result_dict in redacted:
        for dimension in result_dict.get("dimension_results") or []:
            for metric in dimension.get("metric_scores") or []:
                reasoning = metric.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    try:
                        metric["reasoning"] = redact_for_storage(reasoning)
                    except Exception:  # noqa: BLE001 — never break persistence
                        pass
    return redacted


def _metrics_to_metric_set(metrics: List[str]) -> Dict[str, List[str]]:
    """
    Resolve a flat list of metric names into a dimension-keyed metric_set
    using the registry.

    Args:
        metrics: Flat list of metric names (e.g. ["answer_relevance", "agent_utilization"])

    Returns:
        Dict mapping dimension names to lists of metric names,
        e.g. {"Response Quality": ["answer_relevance"], "Multi-Agent": ["agent_utilization"]}

    Raises:
        ValueError: If any metric name is not found in the registry
    """
    from uaef.metrics.registry import get_registry
    registry = get_registry()

    metric_set: Dict[str, List[str]] = {}
    for metric_name in metrics:
        try:
            info = registry.get_metric_info(metric_name)
        except KeyError:
            raise ValueError(
                f"Unknown metric '{metric_name}'. "
                f"Available: {registry.list_metrics()}"
            )
        dimension = info["dimension"] or "unknown"
        metric_set.setdefault(dimension, []).append(metric_name)

    return metric_set


def _select_evaluator(trace, metric_set=None):
    """
    Auto-select the appropriate evaluator based on trace type.

    - MultiAgentTrace → MultiAgentEvaluator (handles multi-agent + all other dimensions)
    - Otherwise → SingleAgentEvaluator (handles all dimensions including multi-turn)
    """
    if isinstance(trace, MultiAgentTrace):
        logger.debug("Auto-selected MultiAgentEvaluator (MultiAgentTrace detected)")
        return MultiAgentEvaluator()

    logger.debug("Auto-selected SingleAgentEvaluator")
    return SingleAgentEvaluator()


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate(
    trace: Union[AgentTrace, MultiAgentTrace, Dict[str, Any]],
    ground_truth: Optional[Union[GroundTruth, Dict[str, Any]]] = None,
    adapter: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    metric_set: Optional[Dict[str, List[str]]] = None,
    dimension_weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    context: Optional[List[str]] = None,
    persist: bool = False,
    experiment_name: Optional[str] = None,
    experiment_objective: Optional[str] = None,
    created_by: Optional[str] = None,
    **kwargs
) -> EvaluationResult:
    """
    Evaluate an agent trace with automatic evaluator selection.
    
    Args:
        trace: Agent trace to evaluate (AgentTrace, MultiAgentTrace, or raw output)
        ground_truth: Expected correct outputs (optional)
        adapter: Framework adapter name ("langgraph", "bedrock", etc.)
        metrics: Flat list of metric names. Dimensions resolved automatically.
        metric_set: Dict mapping dimension names to metric name lists.
        dimension_weights: Weights for each dimension (uses defaults if None)
        thresholds: Threshold values for pass/fail (uses defaults if None)
        context: List of context documents (optional)
        persist: If True, creates a new experiment and persists to DynamoDB + S3.
        experiment_name: Name for the new experiment (defaults to "default").
        experiment_objective: Description/objective for the experiment.
        created_by: Security review H-02: identity (e.g. Cognito ``sub``) of
            the caller, recorded on the persisted experiment when
            ``persist=True`` so a service in front of the library (see
            ``uaef-service``) can scope experiment listing/access to the
            owner. Ignored when ``persist=False``. Optional and unused by
            direct library callers with no per-caller identity concept.
        **kwargs: Additional evaluation parameters
    
    Returns:
        EvaluationResult with experiment_id populated when persist=True.
    
    Examples:
        Ephemeral (no persistence):
            >>> result = evaluate(trace=trace, ground_truth=gt)
        
        Persisted:
            >>> result = evaluate(
            ...     trace=trace,
            ...     persist=True,
            ...     experiment_name="Customer Support Agent v2",
            ...     experiment_objective="Measure quality after prompt changes",
            ... )
    """
    logger.info("Starting evaluation")
    
    # Transform trace if adapter is specified
    if adapter is not None:
        logger.debug(f"Using adapter: {adapter}")
        adapter_instance = get_adapter(adapter)
        trace = adapter_instance.transform_to_canonical(trace)

    # Validate input type. evaluate() handles a single trace or a single multi-turn
    # session dict — multi-session lists must be passed to batch_evaluate() explicitly.
    if isinstance(trace, list):
        raise ValueError(
            "evaluate() does not accept a list of sessions. "
            "Use batch_evaluate() for multi-session input."
        )
    if not isinstance(trace, (AgentTrace, MultiAgentTrace, dict)):
        raise ValueError(
            "trace must be an AgentTrace, MultiAgentTrace, or session dict "
            "({'per_turn_traces': [...], 'full_trace': ...})"
        )

    # Transform ground_truth if it's a dict
    if ground_truth is not None and not isinstance(ground_truth, GroundTruth):
        ground_truth = GroundTruth(**ground_truth)
    
    # If flat metrics list provided, resolve to metric_set (dimension → metrics)
    if metrics is not None:
        metric_set = _metrics_to_metric_set(metrics)
        logger.debug(f"Resolved {len(metrics)} metrics into {len(metric_set)} dimensions: {metric_set}")

    # Multi-turn session dict: dispatch to the per-turn evaluator's evaluate_multi_turn
    if isinstance(trace, dict):
        if "per_turn_traces" not in trace or "full_trace" not in trace:
            raise ValueError(
                "Session dict must contain both 'per_turn_traces' and 'full_trace' keys"
            )
        evaluator = _select_evaluator(trace["full_trace"], metric_set)
        result = evaluator.evaluate_multi_turn(
            per_turn_traces=trace["per_turn_traces"],
            full_trace=trace["full_trace"],
            ground_truth=ground_truth,
            metric_set=metric_set,
            dimension_weights=dimension_weights,
            thresholds=thresholds,
            context=context,
            **kwargs,
        )
    else:
        # Single AgentTrace / MultiAgentTrace path
        evaluator = _select_evaluator(trace, metric_set)
        result = evaluator.evaluate(
            trace=trace,
            ground_truth=ground_truth,
            metric_set=metric_set,
            dimension_weights=dimension_weights,
            thresholds=thresholds,
            context=context,
            **kwargs
        )

    logger.info(f"Evaluation complete: score={result.overall_score:.3f}")
    
    # Persist if requested — always creates a new experiment
    if persist:
        from uuid import UUID as _UUID

        from uaef.utils.serialization import serialize_results

        exp_id, exp_name = _resolve_experiment(
            experiment_name=experiment_name,
            experiment_objective=experiment_objective,
        )
        result.experiment_id = _UUID(exp_id)

        store = _get_store()
        result_dicts = serialize_results([result])
        # Security review M-02: opt-in redaction of the persisted copy only
        # (the `result` object returned below to the caller is untouched).
        result_dicts = _redact_result_dicts_for_persistence(result_dicts)

        store.save_experiment(
            experiment_id=exp_id,
            experiment_name=exp_name,
            evaluation_results=result_dicts,
            experiment_objective=experiment_objective,
            created_by=created_by,
        )
        logger.info(f"Persisted evaluation to new experiment={exp_id}")
    
    return result


def batch_evaluate(
    traces: List[Union[AgentTrace, Dict[str, Any]]],
    ground_truths: Optional[List[Union[GroundTruth, Dict[str, Any]]]] = None,
    adapter: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    metric_set: Optional[Dict[str, List[str]]] = None,
    max_workers: int = 4,
    persist: bool = False,
    experiment_name: Optional[str] = None,
    experiment_objective: Optional[str] = None,
    created_by: Optional[str] = None,
    **kwargs
) -> List[EvaluationResult]:
    """
    Evaluate multiple agent traces in batch mode with parallel processing.
    
    Args:
        traces: List of agent traces to evaluate
        ground_truths: List of expected correct outputs (optional)
        adapter: Framework adapter name.
        metrics: Flat list of metric names. Dimensions resolved automatically.
        metric_set: Dict mapping dimension names to metric name lists.
        max_workers: Maximum number of parallel workers
        persist: If True, creates a new experiment and persists all results.
        experiment_name: Name for the new experiment (defaults to "default").
        experiment_objective: Description/objective for the experiment.
        created_by: Security review H-02: identity (e.g. Cognito ``sub``) of
            the caller, recorded on the persisted experiment when
            ``persist=True`` — see ``evaluate()``'s docstring for details.
        **kwargs: Additional evaluation parameters
    
    Returns:
        List of EvaluationResult objects (with experiment_id when persist=True)
    
    Examples:
        Batch evaluate and persist:
            >>> results = batch_evaluate(
            ...     traces=[trace1, trace2, trace3],
            ...     ground_truths=[gt1, gt2, gt3],
            ...     persist=True,
            ...     experiment_name="Batch Regression Test",
            ... )
    """
    logger.info(f"Starting batch evaluation of {len(traces)} traces")
    
    # Validate ground_truths length
    if ground_truths is not None and len(ground_truths) != len(traces):
        raise ValueError(
            f"ground_truths length ({len(ground_truths)}) must match "
            f"traces length ({len(traces)})"
        )
    
    # Transform traces if adapter is specified
    if adapter is not None:
        logger.debug(f"Using adapter: {adapter}")
        adapter_instance = get_adapter(adapter)
        traces = [adapter_instance.transform_to_canonical(t) for t in traces]
    
    # Transform ground_truths if needed
    if ground_truths is not None:
        ground_truths = [
            gt if isinstance(gt, GroundTruth) else GroundTruth(**gt)
            for gt in ground_truths
        ]
    
    # If flat metrics list provided, resolve to metric_set (dimension → metrics)
    if metrics is not None and metric_set is None:
        metric_set = _metrics_to_metric_set(metrics)
        logger.debug(f"Resolved {len(metrics)} metrics into {len(metric_set)} dimensions: {metric_set}")

    # Detect multi-turn session dicts in the input list. We route them through
    # SingleAgentEvaluator.evaluate_multi_turn() instead of the OfflineEvaluator.
    def _is_session_dict(t):
        return isinstance(t, dict) and "per_turn_traces" in t and "full_trace" in t

    has_session_dicts = any(_is_session_dict(t) for t in traces)

    if has_session_dicts:
        # Pick the right evaluator per session based on the full trace type so
        # MultiAgentTrace sessions go through MultiAgentEvaluator and AgentTrace
        # sessions go through SingleAgentEvaluator.
        results: List[EvaluationResult] = []
        for i, t in enumerate(traces):
            gt = ground_truths[i] if ground_truths else None
            if _is_session_dict(t):
                evaluator = _select_evaluator(t["full_trace"], metric_set)
                result = evaluator.evaluate_multi_turn(
                    per_turn_traces=t["per_turn_traces"],
                    full_trace=t["full_trace"],
                    ground_truth=gt,
                    metric_set=metric_set,
                    **kwargs,
                )
            else:
                evaluator = _select_evaluator(t, metric_set)
                result = evaluator.evaluate(
                    trace=t,
                    ground_truth=gt,
                    metric_set=metric_set,
                    **kwargs,
                )
            results.append(result)
    else:
        # All-AgentTrace path: use the parallel offline evaluator
        evaluator = OfflineEvaluator(max_workers=max_workers)
        results = evaluator.batch_evaluate(
            traces=traces,
            ground_truths=ground_truths,
            metric_set=metric_set,
            **kwargs
        )

    logger.info(f"Batch evaluation complete: {len(results)} results")
    
    # Persist if requested — always creates a new experiment
    if persist:
        from uuid import UUID as _UUID

        from uaef.utils.serialization import serialize_results

        exp_id, exp_name = _resolve_experiment(
            experiment_name=experiment_name,
            experiment_objective=experiment_objective,
        )
        for result in results:
            result.experiment_id = _UUID(exp_id)

        store = _get_store()
        result_dicts = serialize_results(results)
        # Security review M-02: opt-in redaction of the persisted copy only
        # (the `results` list returned below to the caller is untouched).
        result_dicts = _redact_result_dicts_for_persistence(result_dicts)

        store.save_experiment(
            experiment_id=exp_id,
            experiment_name=exp_name,
            evaluation_results=result_dicts,
            experiment_objective=experiment_objective,
            created_by=created_by,
        )
        logger.info(
            f"Persisted {len(results)} evaluations to new experiment={exp_id}"
        )
    
    return results



# ============================================================================
# Experiment Management Functions
# ============================================================================

def create_experiment(
    name: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Experiment:
    """
    Create a new experiment for tracking agent evaluations.
    
    Args:
        name: Human-readable experiment name
        description: Detailed description of the experiment
        metadata: Additional metadata (tags, owner, project, etc.)
    
    Returns:
        Created Experiment instance
    
    Examples:
        Create a simple experiment:
            >>> experiment = create_experiment(
            ...     name="Customer Support Agent v2",
            ...     description="Testing improved tool selection"
            ... )
            >>> print(f"Experiment ID: {experiment.experiment_id}")
        
        Create with metadata:
            >>> experiment = create_experiment(
            ...     name="Production Agent",
            ...     description="Final production version",
            ...     metadata={"owner": "ml-team", "tags": ["production", "v2"]}
            ... )
    """
    logger.info(f"Creating experiment: {name}")
    return _experiment_manager.create_experiment(
        name=name,
        description=description,
        metadata=metadata
    )


def create_run(
    experiment_id: UUID,
    run_name: str,
    agent_design: Union[AgentDesign, Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    git_commit: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ExperimentRun:
    """
    Create a new run within an experiment.
    
    Args:
        experiment_id: ID of the parent experiment
        run_name: Human-readable name for this run
        agent_design: Complete agent design configuration (AgentDesign or dict)
        config: Full configuration snapshot for reproducibility
        git_commit: Git commit hash for code versioning
        metadata: Additional run metadata
    
    Returns:
        Created ExperimentRun instance
    
    Raises:
        ValueError: If experiment_id doesn't exist
    
    Examples:
        Create a run with dict config:
            >>> run = create_run(
            ...     experiment_id=experiment.experiment_id,
            ...     run_name="baseline-v1",
            ...     agent_design={
            ...         "framework": "langgraph",
            ...         "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            ...         "tools": ["search", "calculator"]
            ...     },
            ...     config={"metric_set": ["tool_accuracy"]},
            ...     git_commit="a1b2c3d4"
            ... )
        
        Create with AgentDesign object:
            >>> design = AgentDesign(
            ...     framework="bedrock",
            ...     model="anthropic.claude-3-sonnet-20240229-v1:0",
            ...     tools=["search"]
            ... )
            >>> run = create_run(
            ...     experiment_id=experiment.experiment_id,
            ...     run_name="improved-v2",
            ...     agent_design=design
            ... )
    """
    logger.info(f"Creating run: {run_name} for experiment {experiment_id}")
    
    # Convert dict to AgentDesign if needed
    if not isinstance(agent_design, AgentDesign):
        agent_design = AgentDesign(**agent_design)
    
    return _experiment_manager.create_run(
        experiment_id=experiment_id,
        run_name=run_name,
        agent_design=agent_design,
        config=config or {},
        git_commit=git_commit,
        metadata=metadata
    )


def record_evaluation(
    run_id: UUID,
    evaluation_result: EvaluationResult
) -> None:
    """
    Record an evaluation result for a run.
    
    Args:
        run_id: ID of the run
        evaluation_result: Evaluation result to record
    
    Raises:
        ValueError: If run_id doesn't exist
    
    Examples:
        Record evaluation result:
            >>> result = evaluate(trace, ground_truth, adapter="langgraph")
            >>> record_evaluation(run.run_id, result)
    """
    logger.debug(f"Recording evaluation for run {run_id}")
    _experiment_manager.record_evaluation(run_id, evaluation_result)


def get_run_evaluations(run_id: UUID) -> List[EvaluationResult]:
    """
    Get all evaluation results for a run.
    
    Args:
        run_id: ID of the run
    
    Returns:
        List of EvaluationResult instances for the run
    
    Raises:
        ValueError: If run doesn't exist
    
    Examples:
        Get evaluations for a run:
            >>> evaluations = get_run_evaluations(run.run_id)
            >>> print(f"Found {len(evaluations)} evaluations")
    """
    logger.debug(f"Getting evaluations for run {run_id}")
    return _experiment_manager.get_evaluations(run_id)


def compare_runs(
    baseline_run_id: UUID,
    new_run_id: UUID,
    threshold: float = 5.0
) -> ComparisonReport:
    """
    Compare two experiment runs to identify regressions and improvements.
    
    Args:
        baseline_run_id: ID of the baseline run
        new_run_id: ID of the new run to compare
        threshold: Percentage threshold for detecting changes (default: 5.0%)
    
    Returns:
        ComparisonReport with detailed comparison results
    
    Raises:
        ValueError: If run IDs don't exist
    
    Examples:
        Compare two runs:
            >>> comparison = compare_runs(
            ...     baseline_run_id=run1.run_id,
            ...     new_run_id=run2.run_id,
            ...     threshold=5.0
            ... )
            >>> print(f"Regressions: {comparison.regressions}")
            >>> print(f"Improvements: {comparison.improvements}")
            >>> 
            >>> # Print detailed comparison
            >>> for mc in comparison.metric_comparisons:
            ...     print(f"{mc.metric_name}: {mc.baseline_score:.2f} → {mc.current_score:.2f}")
    """
    logger.info(f"Comparing runs: {baseline_run_id} vs {new_run_id}")
    
    # Get runs
    baseline_run = _experiment_manager.get_run(baseline_run_id)
    new_run = _experiment_manager.get_run(new_run_id)
    
    if baseline_run is None:
        raise ValueError(f"Baseline run {baseline_run_id} not found")
    if new_run is None:
        raise ValueError(f"New run {new_run_id} not found")
    
    # Compare
    return _comparison_engine.compare_runs(
        baseline_run=baseline_run,
        current_run=new_run,
        regression_threshold=threshold,
        improvement_threshold=threshold
    )


def detect_regressions(
    baseline_run_id: UUID,
    new_run_id: UUID,
    threshold: float = 5.0
) -> RegressionReport:
    """
    Detect regressions between two experiment runs.
    
    Args:
        baseline_run_id: ID of the baseline run
        new_run_id: ID of the new run to check
        threshold: Percentage threshold for regression detection (default: 5.0%)
    
    Returns:
        RegressionReport with detected regressions and severity
    
    Examples:
        Detect regressions:
            >>> report = detect_regressions(
            ...     baseline_run_id=run1.run_id,
            ...     new_run_id=run2.run_id,
            ...     threshold=5.0
            ... )
            >>> if report.has_regressions:
            ...     print(f"Found {len(report.regressions)} regressions!")
            ...     for reg in report.regressions:
            ...         print(f"  {reg.metric_name}: {reg.severity}")
    """
    logger.info(f"Detecting regressions: {baseline_run_id} vs {new_run_id}")
    
    # Get runs
    baseline_run = _experiment_manager.get_run(baseline_run_id)
    new_run = _experiment_manager.get_run(new_run_id)
    
    if baseline_run is None:
        raise ValueError(f"Baseline run {baseline_run_id} not found")
    if new_run is None:
        raise ValueError(f"New run {new_run_id} not found")
    
    # Detect regressions
    return _regression_detector.detect_regressions(
        baseline_run=baseline_run,
        current_run=new_run,
        threshold=threshold
    )



# ============================================================================
# Simulation Functions (PLACEHOLDER)
# ============================================================================

def generate_conversation(
    scenario_config: Dict[str, Any],
    user_model: Dict[str, Any],
    agent_config: Dict[str, Any],
    max_turns: int = 10
) -> Dict[str, Any]:
    """
    Generate a simulated conversation (PLACEHOLDER).
    
    NOTE: This is a placeholder implementation. Full conversation simulation
    will be implemented in future work.
    
    Args:
        scenario_config: Scenario configuration (domain, goal, success criteria)
        user_model: User behavior model (personality, patience, knowledge)
        agent_config: Agent configuration (framework, model, tools)
        max_turns: Maximum number of conversation turns
    
    Returns:
        Generated conversation with messages and metadata
    
    Raises:
        NotImplementedError: This feature is not yet implemented
    
    Examples:
        Generate conversation (future):
            >>> conversation = generate_conversation(
            ...     scenario_config={
            ...         "domain": "customer_support",
            ...         "user_goal": "Get refund",
            ...         "success_criteria": ["Refund approved"]
            ...     },
            ...     user_model={
            ...         "personality": "frustrated",
            ...         "patience_level": 3
            ...     },
            ...     agent_config={
            ...         "framework": "langgraph",
            ...         "model": "claude-3-sonnet"
            ...     },
            ...     max_turns=10
            ... )
    """
    raise NotImplementedError(
        "Conversation simulation is not yet implemented. "
        "This is a placeholder for future functionality."
    )


def run_scenario(
    scenario_id: str,
    agent_config: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run a predefined scenario (PLACEHOLDER).
    
    NOTE: This is a placeholder implementation. Full scenario execution
    will be implemented in future work.
    
    Args:
        scenario_id: ID of the predefined scenario
        agent_config: Agent configuration
        parameters: Optional scenario parameters
    
    Returns:
        Scenario execution results
    
    Raises:
        NotImplementedError: This feature is not yet implemented
    
    Examples:
        Run scenario (future):
            >>> result = run_scenario(
            ...     scenario_id="customer_support_refund",
            ...     agent_config={"framework": "langgraph", "model": "gpt-4"}
            ... )
    """
    raise NotImplementedError(
        "Scenario execution is not yet implemented. "
        "This is a placeholder for future functionality."
    )



# ============================================================================
# HITL (Human-in-the-Loop) Functions
# ============================================================================

def add_to_review_queue(
    item_type: Union[ItemType, str],
    content: Dict[str, Any],
    confidence_score: float,
    llm_scores: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ReviewItem:
    """
    Add an item to the human review queue.
    
    Args:
        item_type: Type of item ("llm_evaluation", "agent_response", "simulated_conversation")
        content: Item content (question, response, context, etc.)
        confidence_score: Confidence score (0-1) for automatic priority assignment
        llm_scores: Optional LLM-generated scores
        metadata: Optional additional metadata
    
    Returns:
        Created ReviewItem with assigned priority
    
    Examples:
        Add LLM evaluation for review:
            >>> item = add_to_review_queue(
            ...     item_type="llm_evaluation",
            ...     content={
            ...         "question": "What is AI?",
            ...         "response": "AI is artificial intelligence...",
            ...         "context": ["AI definition document"]
            ...     },
            ...     confidence_score=0.65,
            ...     llm_scores={"answer_relevance": 0.75}
            ... )
            >>> print(f"Priority: {item.priority}")
        
        Add agent response for review:
            >>> item = add_to_review_queue(
            ...     item_type="agent_response",
            ...     content={"response": "Here's your answer..."},
            ...     confidence_score=0.45  # Low confidence -> high priority
            ... )
    """
    logger.debug(f"Adding item to review queue: type={item_type}")
    
    # Convert string to ItemType enum if needed
    if isinstance(item_type, str):
        item_type = ItemType(item_type)
    
    # Create review item
    item = ReviewItem(
        item_type=item_type,
        content=content,
        confidence_score=confidence_score,
        llm_scores=llm_scores or {},
        metadata=metadata or {}
    )
    
    # Add to queue with automatic priority assignment
    return _review_queue.add_to_queue(item, auto_priority=True)


def submit_review(
    item_id: UUID,
    human_scores: Dict[str, float],
    reviewer_id: str,
    feedback: Optional[str] = None
) -> ReviewItem:
    """
    Submit a human review for an item in the queue.
    
    Args:
        item_id: UUID of the item to review
        human_scores: Dictionary of metric_name -> score (0-1)
        reviewer_id: ID of the human reviewer
        feedback: Optional textual feedback
    
    Returns:
        Updated ReviewItem with completed status
    
    Raises:
        ValueError: If item not found or already completed
    
    Examples:
        Submit review:
            >>> reviewed = submit_review(
            ...     item_id=item.item_id,
            ...     human_scores={
            ...         "answer_relevance": 0.85,
            ...         "answer_correctness": 0.90
            ...     },
            ...     reviewer_id="reviewer_123",
            ...     feedback="Good response but could be more detailed"
            ... )
            >>> print(f"Status: {reviewed.status}")
    """
    logger.debug(f"Submitting review for item {item_id}")
    return _review_queue.submit_review(
        item_id=item_id,
        human_scores=human_scores,
        reviewer_id=reviewer_id,
        feedback=feedback
    )


def calculate_agreement(
    item_ids: List[UUID],
    include_inter_rater: bool = True,
    include_llm_human: bool = True
) -> AgreementMetrics:
    """
    Calculate agreement metrics for reviewed items.
    
    Args:
        item_ids: List of item IDs to calculate agreement for
        include_inter_rater: Calculate inter-rater agreement between humans
        include_llm_human: Calculate LLM-human correlation
    
    Returns:
        AgreementMetrics with correlation scores and statistics
    
    Examples:
        Calculate agreement:
            >>> metrics = calculate_agreement(
            ...     item_ids=[item1.item_id, item2.item_id, item3.item_id],
            ...     include_inter_rater=True,
            ...     include_llm_human=True
            ... )
            >>> print(f"LLM-Human correlation: {metrics.llm_human_correlation:.2f}")
            >>> print(f"Inter-rater agreement: {metrics.inter_rater_agreement:.2f}")
    """
    logger.info(f"Calculating agreement for {len(item_ids)} items")
    
    # Get items from queue
    items = [_review_queue.get_item(item_id) for item_id in item_ids]
    
    # Calculate agreement
    return _agreement_calculator.calculate_agreement(
        items=items,
        include_inter_rater=include_inter_rater,
        include_llm_human=include_llm_human
    )



# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Evaluation functions
    "evaluate",
    "batch_evaluate",
    # Metric catalog
    "get_full_metric_catalog",
    # Experiment management functions
    "create_experiment",
    "create_run",
    "record_evaluation",
    "get_run_evaluations",
    "compare_runs",
    "detect_regressions",
    # Simulation functions (placeholder)
    "generate_conversation",
    "run_scenario",
    # HITL functions
    "add_to_review_queue",
    "submit_review",
    "calculate_agreement",
    # Re-export key types for convenience
    "AgentTrace",
    "GroundTruth",
    "EvaluationResult",
    "Experiment",
    "ExperimentRun",
    "AgentDesign",
    "ComparisonReport",
    "RegressionReport",
    "ReviewItem",
    "ItemType",
    "AgreementMetrics",
]
