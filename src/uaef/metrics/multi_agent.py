# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-agent coordination metrics for UAEF.

This module implements metrics for evaluating multi-agent coordination including
agent utilization, delegation quality, workflow completion, and coordination efficiency.
"""

import json
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError

from uaef.config import get_config
from uaef.metrics.base import BaseMetric
from uaef.metrics.utils import (
    build_claude_judge_body,
    build_judge_system_prompt,
    extract_json_from_llm_response,
    validate_judge_response,
    wrap_untrusted,
)
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.metric_score import MetricScore
from uaef.models.multi_agent_trace import MultiAgentTrace, WorkflowStatus


class AgentUtilizationMetric(BaseMetric):
    """
    Metric for evaluating agent utilization (workload distribution).
    
    Measures how evenly work is distributed across agents. Identifies
    underutilized or overutilized agents. This is a deterministic metric.
    """
    
    def get_name(self) -> str:
        """Get metric name."""
        return "agent_utilization"
    
    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False
    
    def requires_llm_judge(self) -> bool:
        """This metric does not require LLM judge."""
        return False
    
    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates workload distribution across agents"
    
    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Multi-Agent"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate agent utilization score."""
        # Check if this is a multi-agent trace
        if not isinstance(evaluation_input.trace, MultiAgentTrace):
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: not a multi-agent trace",
                metadata={"warning": "not_applicable"}
            )

        multi_trace: MultiAgentTrace = evaluation_input.trace
        
        if len(multi_trace.agent_traces) == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no agent traces",
                metadata={"warning": "not_applicable", "agent_count": 0}
            )
        
        # Calculate utilization metrics for each agent
        agent_metrics = {}
        for agent_id, trace in multi_trace.agent_traces.items():
            # Count messages and tool calls as indicators of work
            message_count = len(trace.messages)
            tool_call_count = len(trace.tool_calls)
            total_work = message_count + tool_call_count
            
            agent_metrics[agent_id] = {
                "message_count": message_count,
                "tool_call_count": tool_call_count,
                "total_work": total_work
            }
        
        # Calculate distribution statistics
        work_values = [m["total_work"] for m in agent_metrics.values()]
        total_work = sum(work_values)
        
        if total_work == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=1.0,
                reasoning="No work performed by any agent",
                metadata={"agent_count": len(multi_trace.agent_traces)}
            )
        
        # Calculate coefficient of variation (CV) to measure distribution evenness
        # Lower CV = more even distribution = higher score
        mean_work = total_work / len(work_values)
        variance = sum((w - mean_work) ** 2 for w in work_values) / len(work_values)
        std_dev = variance ** 0.5
        cv = std_dev / mean_work if mean_work > 0 else 0
        
        # Convert CV to score (0 CV = perfect distribution = score 1.0)
        # Higher CV = worse distribution = lower score
        # Use exponential decay: score = exp(-cv)
        import math
        score = math.exp(-cv)
        score = max(0.0, min(1.0, score))
        
        # Identify underutilized and overutilized agents
        underutilized = [aid for aid, m in agent_metrics.items() if m["total_work"] < mean_work * 0.5]
        overutilized = [aid for aid, m in agent_metrics.items() if m["total_work"] > mean_work * 1.5]
        
        reasoning_parts = [f"Work distribution CV: {cv:.2f}"]
        if underutilized:
            reasoning_parts.append(f"Underutilized: {', '.join(underutilized)}")
        if overutilized:
            reasoning_parts.append(f"Overutilized: {', '.join(overutilized)}")
        
        reasoning = "; ".join(reasoning_parts)
        
        return MetricScore(
            metric_name=self.get_name(),
            score=score,
            reasoning=reasoning,
            metadata={
                "agent_count": len(multi_trace.agent_traces),
                "total_work": total_work,
                "mean_work": mean_work,
                "cv": cv,
                "agent_metrics": agent_metrics,
                "underutilized": underutilized,
                "overutilized": overutilized
            }
        )

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Async version (just calls synchronous version for deterministic metric)."""
        return self.calculate(evaluation_input)

    @staticmethod
    def calculate_batch(evaluation_inputs: List[EvaluationInput]) -> MetricScore:
        """
        Calculate agent utilization across a batch of multi-agent traces.

        Batch calculation is more meaningful than single-trace calculation because:
        - Some tasks naturally require more agents, some require fewer
        - Single-trace utilization depends heavily on task complexity
        - Batch metrics show overall agent usage patterns across diverse tasks

        Args:
            evaluation_inputs: List of evaluation inputs with multi_agent_traces

        Returns:
            MetricScore with batch-level agent utilization across all traces
        """
        # Filter to only multi-agent traces
        valid_traces = []
        for eval_input in evaluation_inputs:
            if isinstance(eval_input.trace, MultiAgentTrace):
                valid_traces.append(eval_input.trace)

        if not valid_traces:
            return MetricScore(
                metric_name="agent_utilization_batch",
                score=None,
                reasoning="Cannot evaluate: no multi-agent traces in batch",
                metadata={"warning": "not_applicable", "batch_size": len(evaluation_inputs)}
            )

        # Aggregate agent work across all traces
        agent_work_across_traces = {}

        for multi_trace in valid_traces:
            for agent_id, trace in multi_trace.agent_traces.items():
                if agent_id not in agent_work_across_traces:
                    agent_work_across_traces[agent_id] = []

                # Count work for this agent in this trace
                work = len(trace.messages) + len(trace.tool_calls)
                agent_work_across_traces[agent_id].append(work)

        # Calculate per-agent average work across traces
        agent_avg_work = {}
        for agent_id, work_list in agent_work_across_traces.items():
            agent_avg_work[agent_id] = sum(work_list) / len(work_list)

        # Calculate distribution metrics
        avg_values = list(agent_avg_work.values())
        total_work = sum(avg_values)
        total_agents = len(agent_avg_work)

        if total_work == 0:
            return MetricScore(
                metric_name="agent_utilization_batch",
                score=1.0,
                reasoning=f"No work performed across {len(valid_traces)} traces",
                metadata={
                    "batch_size": len(valid_traces),
                    "agent_count": total_agents
                }
            )

        mean_work = total_work / len(avg_values)
        variance = sum((w - mean_work) ** 2 for w in avg_values) / len(avg_values)
        std_dev = variance ** 0.5
        cv = std_dev / mean_work if mean_work > 0 else 0

        # Score based on CV
        import math
        score = math.exp(-cv)
        score = max(0.0, min(1.0, score))

        # Identify utilization patterns
        underutilized = [aid for aid, w in agent_avg_work.items() if w < mean_work * 0.5]
        overutilized = [aid for aid, w in agent_avg_work.items() if w > mean_work * 1.5]

        reasoning = f"Batch utilization across {len(valid_traces)} traces: CV={cv:.2f}, {total_agents} agents"
        if underutilized:
            reasoning += f", {len(underutilized)} underutilized"
        if overutilized:
            reasoning += f", {len(overutilized)} overutilized"

        return MetricScore(
            metric_name="agent_utilization_batch",
            score=score,
            reasoning=reasoning,
            metadata={
                "batch_size": len(valid_traces),
                "agent_count": total_agents,
                "mean_work": mean_work,
                "cv": cv,
                "agent_avg_work": agent_avg_work,
                "underutilized": underutilized,
                "overutilized": overutilized
            }
        )


class DelegationQualityMetric(BaseMetric):
    """
    Metric for evaluating delegation quality (task assignment appropriateness).
    
    Evaluates whether tasks were assigned to appropriate agents based on
    their capabilities. Uses LLM judge to assess delegation decisions.
    """
    
    def get_name(self) -> str:
        """Get metric name."""
        return "delegation_quality"
    
    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False
    
    def requires_llm_judge(self) -> bool:
        """This metric requires LLM judge."""
        return True
    
    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates whether tasks were assigned to appropriate agents"
    
    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Multi-Agent"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate delegation quality score."""
        # Check if this is a multi-agent trace
        if not isinstance(evaluation_input.trace, MultiAgentTrace):
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: not a multi-agent trace",
                metadata={"warning": "not_applicable"}
            )
        
        multi_trace: MultiAgentTrace = evaluation_input.trace
        
        if len(multi_trace.agent_traces) == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no agent traces",
                metadata={"warning": "missing_data", "agent_count": 0}
            )

        if len(multi_trace.coordination_events) == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no coordination events",
                metadata={"warning": "not_applicable", "event_count": 0}
            )
        
        # Build coordination summary
        coordination_summary = "\n".join([
            f"[{event.timestamp}] {event.from_agent} -> {event.to_agent}: {event.event_type} - {event.message}"
            for event in multi_trace.coordination_events
        ])
        
        # Build agent capabilities summary
        agent_summary = "\n".join([
            f"Agent {agent_id}: {len(trace.messages)} messages, {len(trace.tool_calls)} tool calls"
            for agent_id, trace in multi_trace.agent_traces.items()
        ])
        
        # Use LLM judge to evaluate delegation quality
        try:
            config = get_config()
            score, reasoning = self._llm_judge_delegation(
                coordination_summary=coordination_summary,
                agent_summary=agent_summary,
                judge_id=config.llm_judge.model_id,
                max_tokens=config.llm_judge.max_tokens,
                temperature=config.llm_judge.temperature
            )
            
            return MetricScore(
                metric_name=self.get_name(),
                score=score,
                reasoning=reasoning,
                metadata={
                    "agent_count": len(multi_trace.agent_traces),
                    "coordination_event_count": len(multi_trace.coordination_events)
                }
            )
        except ConnectionError as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: Bedrock client initialization failed: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
        except RuntimeError as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: Bedrock API call failed: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
        except ValueError as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: Invalid score from LLM judge: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
        except ClientError as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: Bedrock API error: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
        except Exception as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: LLM judge error: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
    
    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate delegation quality asynchronously."""
        return self.calculate(evaluation_input)
    
    def _llm_judge_delegation(
        self,
        coordination_summary: str,
        agent_summary: str,
        judge_id: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[float, str]:
        """
        Call LLM judge to evaluate delegation quality.
        
        Args:
            coordination_summary: Summary of coordination events
            agent_summary: Summary of agent capabilities
            judge_id: Bedrock model ID
            max_tokens: Maximum tokens for response
            temperature: Temperature for sampling
            
        Returns:
            Tuple of (score, reasoning)
        """
        # Initialize AWS Bedrock client
        try:
            config = get_config()
            bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=config.aws.region,
                aws_access_key_id=config.aws.access_key_id,
                aws_secret_access_key=config.aws.secret_access_key,
                aws_session_token=config.aws.session_token
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Bedrock client: {str(e)}")
        
        # Prepare prompt
        resp_fmt = """{
                       "score":float,
                       "reasoning": str
                   }
               """
        
        # Security review H-01: rubric in system field; agent_summary and
        # coordination_summary are both derived from the agents' own trace
        # activity (agent names, message/tool-call counts, coordination
        # event messages) and are untrusted, so both are wrapped.
        system_rubric = """
            You are a strict AI evaluator that assesses delegation quality in multi-agent systems.
            Evaluate whether tasks were delegated to the MOST APPROPRIATE agent based on each agent's
            name, role, and observed capabilities (the tools it calls). Be critical and penalize mistakes.

            You will be given agent activity summaries and coordination events.

            EVALUATION CRITERIA (apply strictly):

            1. **Agent-task fit**: Was the task sent to the agent best suited for it?
               - Infer each agent's specialty from its name and the tools it uses.
               - If a clearly more suitable agent existed but was idle or underused, this is a MAJOR flaw.
               - Example: sending a database query task to a "formatting_agent" when a "sql_agent" exists is poor delegation.

            2. **Delegation clarity**: Were delegation messages specific and actionable?
               - Vague messages like "do something" or "handle the thing" are a flaw.
               - Good messages specify what to do, on what data, and what output is expected.

            3. **Handoff efficiency**: Was the number of handoffs reasonable?
               - Excessive back-and-forth (ping-pong) between agents for a simple task is a flaw.
               - Each handoff should move the workflow forward, not backward.

            4. **Workflow progression**: Did the delegation follow a logical order?
               - Steps should build on each other, not repeat or contradict.

            SCORING GUIDE (be strict):
            - 0.9-1.0: Excellent — every task routed to the ideal agent, clear messages, minimal handoffs
            - 0.7-0.8: Good — mostly correct routing with minor inefficiencies
            - 0.4-0.6: Mediocre — some tasks misrouted or vague instructions
            - 0.1-0.3: Poor — tasks sent to wrong agents, unclear messages, excessive handoffs
            - 0.0: Terrible — completely inappropriate delegation across the board

            After providing your explanation in the "reasoning" field, score the delegation quality
            on a scale of 0 to 1 in the "score" field.
            Strictly follow the below json format:{resp_fmt}."""

        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=resp_fmt))
        user_message = (
            wrap_untrusted(agent_summary, "candidate_agent_summary")
            + "\n\n"
            + wrap_untrusted(coordination_summary, "candidate_coordination_summary")
        )

        # Prepare request body
        # Claude 3+ rejects requests with both temperature and top_p;
        # temperature alone is sufficient for deterministic judge scoring.
        body = build_claude_judge_body(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Make API call
        try:
            response_obj = bedrock_client.invoke_model(
                modelId=judge_id,
                body=body,
                accept='application/json',
                contentType='application/json'
            )
        except Exception as e:
            raise RuntimeError(f"Bedrock API call failed: {str(e)}")
        
        # Parse response
        response_body = json.loads(response_obj.get('body').read())
        raw_text = response_body.get('content')[0]['text']
        response_json = extract_json_from_llm_response(raw_text)
        # Security review M-05: validate shape/types before use, not just
        # the score's numeric range.
        response_json = validate_judge_response(response_json)
        response_score = response_json["score"]
        response_reasoning = response_json.get("reasoning", "")
        
        return response_score, response_reasoning


class WorkflowCompletionMetric(BaseMetric):
    """
    Metric for evaluating workflow completion (overall goal achievement).
    
    Evaluates whether the overall multi-agent workflow achieved its goal.
    Requires ground truth with success criteria.
    """
    
    def get_name(self) -> str:
        """Get metric name."""
        return "workflow_completion"
    
    def requires_ground_truth(self) -> bool:
        """This metric requires ground truth."""
        return True
    
    def requires_llm_judge(self) -> bool:
        """This metric uses LLM judge for evaluation."""
        return True
    
    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates whether the overall multi-agent workflow achieved its goal"
    
    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Multi-Agent"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate workflow completion score."""
        # Validate ground truth is provided
        success_criteria = (evaluation_input.ground_truth.expected_output
                            if evaluation_input.ground_truth else None)
        if not success_criteria:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no success criteria provided",
                metadata={"warning": "missing_data"}
            )

        # Check if this is a multi-agent trace
        if not isinstance(evaluation_input.trace, MultiAgentTrace):
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: not a multi-agent trace",
                metadata={"warning": "not_applicable"}
            )

        multi_trace: MultiAgentTrace = evaluation_input.trace

        # No agents means no work was done — cannot evaluate
        if len(multi_trace.agent_traces) == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no agent traces",
                metadata={"warning": "missing_data", "agent_count": 0}
            )

        # Check workflow status
        workflow_status = multi_trace.workflow_status

        # Build workflow summary
        workflow_summary = f"Workflow Status: {workflow_status.value}\n"
        workflow_summary += f"Agents: {', '.join(multi_trace.agent_traces.keys())}\n"
        workflow_summary += f"Coordination Events: {len(multi_trace.coordination_events)}\n"

        # Add final outputs from each agent
        for agent_id, trace in multi_trace.agent_traces.items():
            if trace.messages:
                last_msg = trace.messages[-1]
                if last_msg.role == "assistant":
                    workflow_summary += f"\n{agent_id} final output: {last_msg.content[:200]}"

        # Use LLM judge to evaluate completion (even for failed workflows)
        # Failed workflows can still have partial completion
        try:
            config = get_config()
            score, reasoning = self._llm_judge_completion(
                workflow_summary=workflow_summary,
                success_criteria=success_criteria,
                judge_id=config.llm_judge.model_id,
                max_tokens=config.llm_judge.max_tokens,
                temperature=config.llm_judge.temperature
            )

            # Apply penalty for failed workflows (0.5x multiplier)
            if workflow_status == WorkflowStatus.FAILED and score > 0:
                original_score = score
                score = score * 0.5
                reasoning = f"Workflow failed but achieved {original_score:.1%} partial completion (penalized to {score:.1%}). {reasoning}"

            return MetricScore(
                metric_name=self.get_name(),
                score=score,
                reasoning=reasoning,
                metadata={
                    "workflow_status": workflow_status.value,
                    "agent_count": len(multi_trace.agent_traces),
                    "success_criteria": success_criteria[:100],
                    "partial_completion": score > 0 and workflow_status == WorkflowStatus.FAILED
                }
            )
        except ConnectionError as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: Bedrock client initialization failed: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
        except RuntimeError as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: Bedrock API call failed: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
        except ValueError as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: Invalid score from LLM judge: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
        except ClientError as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: Bedrock API error: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
        except Exception as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: LLM judge error: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
    
    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate workflow completion asynchronously."""
        return self.calculate(evaluation_input)
    
    def _llm_judge_completion(
        self,
        workflow_summary: str,
        success_criteria: str,
        judge_id: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[float, str]:
        """
        Call LLM judge to evaluate workflow completion.
        
        Args:
            workflow_summary: Summary of workflow execution
            success_criteria: Expected success criteria from ground truth
            judge_id: Bedrock model ID
            max_tokens: Maximum tokens for response
            temperature: Temperature for sampling
            
        Returns:
            Tuple of (score, reasoning)
        """
        # Initialize AWS Bedrock client
        try:
            config = get_config()
            bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=config.aws.region,
                aws_access_key_id=config.aws.access_key_id,
                aws_secret_access_key=config.aws.secret_access_key,
                aws_session_token=config.aws.session_token
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Bedrock client: {str(e)}")
        
        # Prepare prompt
        resp_fmt = """{
                       "score":float,
                       "reasoning": str
                   }
               """
        
        # Security review H-01: rubric in system field; success_criteria is
        # trusted ground-truth data (from the caller's evaluation_input.
        # ground_truth) and stays as plain text; workflow_summary is derived
        # from agent trace output (final assistant messages per agent) and
        # is untrusted, so it is wrapped.
        system_rubric = """
            You are an AI evaluator that assesses workflow completion in multi-agent systems.
            Please evaluate whether the overall workflow achieved its goal.
            You will be given the workflow summary and success criteria.
            
            In your evaluation, check whether:
            - All success criteria were met
            - The workflow completed successfully
            - The final outputs satisfy the requirements
            - All necessary steps were completed
            - The goal was achieved through agent collaboration
            
            After providing your explanation in the "reasoning" tab, you must score the workflow on a scale of 0 to 1 in the "score" tab,
            where 1 means fully complete (all success criteria met, goal achieved)
            and 0 means incomplete (success criteria not met, goal not achieved).
            Strictly follow the below json format:{resp_fmt}."""

        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=resp_fmt))
        user_message = (
            f"[Success Criteria]\n{success_criteria}\n\n"
            + wrap_untrusted(workflow_summary, "candidate_workflow_summary")
        )

        # Prepare request body
        # Claude 3+ rejects requests with both temperature and top_p;
        # temperature alone is sufficient for deterministic judge scoring.
        body = build_claude_judge_body(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Make API call
        try:
            response_obj = bedrock_client.invoke_model(
                modelId=judge_id,
                body=body,
                accept='application/json',
                contentType='application/json'
            )
        except Exception as e:
            raise RuntimeError(f"Bedrock API call failed: {str(e)}")
        
        # Parse response
        response_body = json.loads(response_obj.get('body').read())
        raw_text = response_body.get('content')[0]['text']
        response_json = extract_json_from_llm_response(raw_text)
        # Security review M-05: validate shape/types before use, not just
        # the score's numeric range.
        response_json = validate_judge_response(response_json)
        response_score = response_json["score"]
        response_reasoning = response_json.get("reasoning", "")
        
        return response_score, response_reasoning


class CoordinationEfficiencyMetric(BaseMetric):
    """
    Metric for evaluating coordination efficiency.

    Measures the efficiency of agent coordination based on the number of
    coordination events relative to work completed. This is a deterministic metric.

    Ideal ratios by workflow type:
    - Hierarchical: 0.05 (centralized orchestrator, minimal coordination)
    - Balanced: 0.10 (default, moderate coordination)
    - Collaborative: 0.20 (peer-to-peer, more coordination)
    - Sequential: 0.25 (many handoffs between agents)
    """

    def __init__(self, ideal_ratio: float = 0.10):
        """
        Initialize coordination efficiency metric.

        Args:
            ideal_ratio: Expected ratio of coordination events to work units.
                        Defaults to 0.10 (1 coordination per 10 work units).
        """
        self.ideal_ratio = ideal_ratio

    def get_name(self) -> str:
        """Get metric name."""
        return "coordination_efficiency"

    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False

    def requires_llm_judge(self) -> bool:
        """This metric does not require LLM judge."""
        return False

    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates efficiency of agent coordination"

    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Multi-Agent"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate coordination efficiency score."""
        # Check if this is a multi-agent trace
        if not isinstance(evaluation_input.trace, MultiAgentTrace):
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: not a multi-agent trace",
                metadata={"warning": "not_applicable"}
            )
        
        multi_trace: MultiAgentTrace = evaluation_input.trace
        
        # Count total work (messages + tool calls)
        total_work = sum(
            len(trace.messages) + len(trace.tool_calls)
            for trace in multi_trace.agent_traces.values()
        )
        
        # Count coordination events
        coordination_count = len(multi_trace.coordination_events)
        
        if total_work == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no work performed by any agent",
                metadata={"warning": "missing_data", "total_work": 0, "coordination_count": coordination_count}
            )
        
        # Calculate coordination overhead ratio
        # Lower ratio = more efficient (less coordination overhead)
        # Ideal ratio is around 0.1 (1 coordination per 10 work units)
        coordination_ratio = coordination_count / total_work
        
        # Convert ratio to score
        # Score = 1.0 if ratio <= ideal_ratio
        # Score decreases as ratio increases
        if coordination_ratio <= self.ideal_ratio:
            score = 1.0
            reasoning = f"Efficient coordination: {coordination_count} events for {total_work} work units (ratio: {coordination_ratio:.2f}, ideal: {self.ideal_ratio:.2f})"
        else:
            # Exponential decay for high coordination overhead
            import math
            score = math.exp(-(coordination_ratio - self.ideal_ratio) * 5)
            score = max(0.0, min(1.0, score))
            reasoning = f"High coordination overhead: {coordination_count} events for {total_work} work units (ratio: {coordination_ratio:.2f}, ideal: {self.ideal_ratio:.2f})"
        
        return MetricScore(
            metric_name=self.get_name(),
            score=score,
            reasoning=reasoning,
            metadata={
                "total_work": total_work,
                "coordination_count": coordination_count,
                "coordination_ratio": coordination_ratio,
                "ideal_ratio": self.ideal_ratio
            }
        )
    
    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Async version (just calls synchronous version for deterministic metric)."""
        return self.calculate(evaluation_input)
