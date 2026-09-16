# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment manager for creating and managing experiments and runs."""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from uaef.experiments.models import AgentDesign, Experiment, ExperimentRun
from uaef.models.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Manager for creating and managing experiments and runs.
    
    Provides methods for:
    - Creating experiments and runs
    - Recording evaluation results
    - Setting baselines
    - Retrieving experiment data
    """
    
    def __init__(self):
        """Initialize the experiment manager."""
        # In-memory storage for now (will be replaced with database in Task 6.8)
        self._experiments: Dict[UUID, Experiment] = {}
        self._runs: Dict[UUID, ExperimentRun] = {}
        self._evaluations: Dict[UUID, List[EvaluationResult]] = {}
        logger.info("ExperimentManager initialized")
    
    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            name: Human-readable experiment name
            description: Detailed description of the experiment
            metadata: Additional metadata (tags, owner, project, etc.)
            
        Returns:
            Created Experiment instance
            
        Example:
            >>> manager = ExperimentManager()
            >>> experiment = manager.create_experiment(
            ...     name="Customer Support Agent v2",
            ...     description="Testing improved tool selection",
            ...     metadata={"owner": "ml-team", "tags": ["production"]}
            ... )
        """
        experiment = Experiment(
            name=name,
            description=description,
            metadata=metadata or {}
        )
        
        self._experiments[experiment.experiment_id] = experiment
        logger.info(
            f"Created experiment '{name}' with ID {experiment.experiment_id}"
        )
        
        return experiment
    
    def create_run(
        self,
        experiment_id: UUID,
        run_name: str,
        agent_design: AgentDesign,
        config: Dict[str, Any],
        git_commit: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExperimentRun:
        """
        Create a new run within an experiment.
        
        Args:
            experiment_id: ID of the parent experiment
            run_name: Human-readable name for this run
            agent_design: Complete agent design configuration
            config: Full configuration snapshot for reproducibility
            git_commit: Git commit hash for code versioning
            metadata: Additional run metadata
            
        Returns:
            Created ExperimentRun instance
            
        Raises:
            ValueError: If experiment_id doesn't exist
            
        Example:
            >>> design = AgentDesign(
            ...     framework="langgraph",
            ...     model="anthropic.claude-3-sonnet-20240229-v1:0",
            ...     tools=["search", "calculator"]
            ... )
            >>> run = manager.create_run(
            ...     experiment_id=experiment.experiment_id,
            ...     run_name="baseline-v1",
            ...     agent_design=design,
            ...     config={"metric_set": ["tool_accuracy"]},
            ...     git_commit="a1b2c3d4..."
            ... )
        """
        # Validate experiment exists
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        run = ExperimentRun(
            experiment_id=experiment_id,
            run_name=run_name,
            agent_design=agent_design,
            config_snapshot=config,
            git_commit=git_commit,
            metadata=metadata or {}
        )
        
        self._runs[run.run_id] = run
        self._evaluations[run.run_id] = []
        
        logger.info(
            f"Created run '{run_name}' with ID {run.run_id} "
            f"for experiment {experiment_id}"
        )
        
        return run
    
    def record_evaluation(
        self,
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
            
        Example:
            >>> manager.record_evaluation(
            ...     run_id=run.run_id,
            ...     evaluation_result=evaluation_result
            ... )
        """
        # Validate run exists
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")
        
        # Add evaluation to storage
        self._evaluations[run_id].append(evaluation_result)
        
        # Update run statistics
        run = self._runs[run_id]
        run.evaluation_count = len(self._evaluations[run_id])
        
        # Update aggregate metrics
        self._update_aggregate_metrics(run_id)
        
        logger.debug(
            f"Recorded evaluation {evaluation_result.evaluation_id} "
            f"for run {run_id}"
        )
    
    def _update_aggregate_metrics(self, run_id: UUID) -> None:
        """
        Update aggregate metrics for a run.
        
        Args:
            run_id: ID of the run to update
        """
        evaluations = self._evaluations[run_id]
        if not evaluations:
            return
        
        run = self._runs[run_id]
        
        # Collect all metric scores across evaluations
        metric_scores: Dict[str, List[float]] = {}
        
        for evaluation in evaluations:
            for dimension_result in evaluation.dimension_results:
                for metric_score in dimension_result.metric_scores:
                    metric_name = metric_score.metric_name
                    if metric_name not in metric_scores:
                        metric_scores[metric_name] = []
                    metric_scores[metric_name].append(metric_score.score)
        
        # Calculate mean for each metric
        run.aggregate_metrics = {
            metric_name: sum(scores) / len(scores)
            for metric_name, scores in metric_scores.items()
        }
        
        logger.debug(f"Updated aggregate metrics for run {run_id}")
    
    def set_baseline(
        self,
        experiment_id: UUID,
        run_id: UUID
    ) -> None:
        """
        Set a run as the baseline for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            run_id: ID of the run to set as baseline
            
        Raises:
            ValueError: If experiment or run doesn't exist, or run doesn't belong to experiment
            
        Example:
            >>> manager.set_baseline(
            ...     experiment_id=experiment.experiment_id,
            ...     run_id=run.run_id
            ... )
        """
        # Validate experiment exists
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Validate run exists
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")
        
        # Validate run belongs to experiment
        run = self._runs[run_id]
        if run.experiment_id != experiment_id:
            raise ValueError(
                f"Run {run_id} does not belong to experiment {experiment_id}"
            )
        
        # Set baseline
        experiment = self._experiments[experiment_id]
        experiment.baseline_run_id = run_id
        
        logger.info(
            f"Set run {run_id} as baseline for experiment {experiment_id}"
        )
    
    def get_experiment(self, experiment_id: UUID) -> Optional[Experiment]:
        """
        Get an experiment by ID.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Experiment instance or None if not found
            
        Example:
            >>> experiment = manager.get_experiment(experiment_id)
        """
        return self._experiments.get(experiment_id)
    
    def get_run(self, run_id: UUID) -> Optional[ExperimentRun]:
        """
        Get a run by ID.
        
        Args:
            run_id: ID of the run
            
        Returns:
            ExperimentRun instance or None if not found
            
        Example:
            >>> run = manager.get_run(run_id)
        """
        return self._runs.get(run_id)
    
    def list_experiments(self) -> List[Experiment]:
        """
        List all experiments.
        
        Returns:
            List of all Experiment instances
            
        Example:
            >>> experiments = manager.list_experiments()
        """
        return list(self._experiments.values())
    
    def list_runs(
        self,
        experiment_id: UUID
    ) -> List[ExperimentRun]:
        """
        List all runs for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            List of ExperimentRun instances for the experiment
            
        Raises:
            ValueError: If experiment doesn't exist
            
        Example:
            >>> runs = manager.list_runs(experiment_id)
        """
        # Validate experiment exists
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Filter runs by experiment_id
        return [
            run for run in self._runs.values()
            if run.experiment_id == experiment_id
        ]
    
    def get_evaluations(
        self,
        run_id: UUID
    ) -> List[EvaluationResult]:
        """
        Get all evaluation results for a run.
        
        Args:
            run_id: ID of the run
            
        Returns:
            List of EvaluationResult instances for the run
            
        Raises:
            ValueError: If run doesn't exist
            
        Example:
            >>> evaluations = manager.get_evaluations(run_id)
        """
        # Validate run exists
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")
        
        return self._evaluations.get(run_id, [])
    
    def get_baseline_run(
        self,
        experiment_id: UUID
    ) -> Optional[ExperimentRun]:
        """
        Get the baseline run for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Baseline ExperimentRun instance or None if no baseline set
            
        Raises:
            ValueError: If experiment doesn't exist
            
        Example:
            >>> baseline = manager.get_baseline_run(experiment_id)
        """
        # Validate experiment exists
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self._experiments[experiment_id]
        if experiment.baseline_run_id is None:
            return None
        
        return self._runs.get(experiment.baseline_run_id)
