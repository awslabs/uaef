# Human-in-the-Loop (HITL) Workflow Guide

## Overview

UAEF's Human-in-the-Loop (HITL) system enables you to incorporate human judgment into your evaluation workflow. This guide covers three distinct HITL workflows: LLM judge calibration, online evaluation feedback, and conversation scoring.

## Why Use HITL?

HITL workflows provide:

- **LLM Judge Calibration**: Improve automated evaluation accuracy
- **Quality Assurance**: Validate automated scores with human judgment
- **Edge Case Handling**: Human review for low-confidence evaluations
- **Continuous Improvement**: Use feedback to refine evaluation criteria
- **Trust Building**: Demonstrate evaluation reliability to stakeholders

## Three HITL Workflows

### 1. LLM Judge Calibration

Improve LLM-based metrics by comparing with human judgments:

```
┌─────────────────┐
│ LLM Evaluation  │
│ (Low Confidence)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Human Review    │
│ & Scoring       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Calculate       │
│ Agreement       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Calibrate       │
│ LLM Prompts     │
└─────────────────┘
```

### 2. Online Evaluation Feedback

Collect real-time feedback on agent responses:

```
┌─────────────────┐
│ Agent Response  │
│ (Production)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ User Feedback   │
│ (Thumbs/Rating) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Aggregate       │
│ Feedback        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Identify Issues │
│ & Trends        │
└─────────────────┘
```

### 3. Conversation Scoring

Validate simulated conversations:

```
┌─────────────────┐
│ Simulated       │
│ Conversation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Human Review    │
│ & Quality Score │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Improve         │
│ Simulator       │
└─────────────────┘
```

## Workflow 1: LLM Judge Calibration

### Overview

Calibrate LLM judges by comparing their scores with human judgments, then adjust prompts to improve agreement.

### Step 1: Initialize Calibration Workflow

```python
from uaef.hitl import JudgeCalibrationWorkflow
from uaef.llm_judge import LLMJudge

# Initialize workflow
workflow = JudgeCalibrationWorkflow(
    llm_judge=LLMJudge(),
    confidence_threshold=0.7,  # Queue items below this confidence
    sample_rate=0.1            # Also sample 10% of high-confidence items
)
```

### Step 2: Run Evaluations and Queue Low-Confidence Items

```python
from uaef.evaluation import SingleAgentEvaluator
from uaef.models import EvaluationInput

# Create evaluator
evaluator = SingleAgentEvaluator(
    metrics=["answer_relevance", "safety_score", "hallucination_score"]
)

# Evaluate and queue for review
for test_case in test_cases:
    trace = run_agent(test_case)
    evaluation_input = EvaluationInput(
        trace=trace,
        ground_truth=test_case.ground_truth
    )
    
    # Evaluate
    result = evaluator.evaluate(evaluation_input)
    
    # Check if needs human review
    for dimension in result.dimension_results:
        for metric in dimension.metric_scores:
            if workflow.needs_review(metric):
                # Add to review queue
                workflow.queue_for_review(
                    evaluation_input=evaluation_input,
                    metric_score=metric,
                    priority="high" if metric.score < 0.5 else "normal"
                )

print(f"Queued {workflow.queue_size()} items for review")
```

### Step 3: Human Review Interface

```python
# Get next item to review
review_item = workflow.get_next_review_item()

if review_item:
    print(f"\n=== Review Item {review_item.item_id} ===")
    print(f"Metric: {review_item.metric_name}")
    print(f"LLM Score: {review_item.llm_score:.2f}")
    print(f"LLM Reasoning: {review_item.llm_reasoning}")
    print(f"\nQuestion: {review_item.question}")
    print(f"Answer: {review_item.answer}")
    
    # Collect human score
    human_score = float(input("\nYour score (0.0-1.0): "))
    human_reasoning = input("Your reasoning: ")
    
    # Submit review
    workflow.submit_review(
        item_id=review_item.item_id,
        human_score=human_score,
        human_reasoning=human_reasoning,
        reviewer_id="reviewer_123"
    )
```

### Step 4: Calculate Agreement

```python
from uaef.hitl import AgreementCalculator

calculator = AgreementCalculator()

# Calculate agreement metrics
agreement_metrics = calculator.calculate_llm_human_agreement(
    workflow.get_reviewed_items()
)

print("\n=== Agreement Metrics ===")
print(f"Pearson Correlation: {agreement_metrics.pearson_correlation:.3f}")
print(f"Spearman Correlation: {agreement_metrics.spearman_correlation:.3f}")
print(f"Mean Absolute Error: {agreement_metrics.mean_absolute_error:.3f}")
print(f"Agreement Rate (±0.1): {agreement_metrics.agreement_rate:.1%}")

# Identify disputed items
disputed_items = calculator.identify_disputes(
    workflow.get_reviewed_items(),
    threshold=0.3  # Difference > 0.3 is a dispute
)

print(f"\nDisputed items: {len(disputed_items)}")
for item in disputed_items[:5]:
    print(f"  - {item.metric_name}: LLM={item.llm_score:.2f}, "
          f"Human={item.human_score:.2f}, Diff={abs(item.llm_score - item.human_score):.2f}")
```

### Step 5: Calibrate LLM Prompts

```python
# Analyze disagreements
calibration_report = workflow.analyze_disagreements()

print("\n=== Calibration Insights ===")
print(f"Total reviews: {calibration_report['total_reviews']}")
print(f"High agreement: {calibration_report['high_agreement_count']}")
print(f"Low agreement: {calibration_report['low_agreement_count']}")

# Get prompt improvement suggestions
suggestions = workflow.get_prompt_suggestions()

for suggestion in suggestions:
    print(f"\nMetric: {suggestion.metric_name}")
    print(f"Issue: {suggestion.issue_description}")
    print(f"Suggested change: {suggestion.suggested_prompt_change}")
    print(f"Expected improvement: {suggestion.expected_improvement}")

# Apply calibration (update prompts)
workflow.apply_calibration(
    metric_name="answer_relevance",
    new_prompt=suggestions[0].suggested_prompt_change
)
```

### Step 6: Create Benchmark

```python
from uaef.hitl import BenchmarkVersioning

benchmark = BenchmarkVersioning()

# Create benchmark from reviewed items
benchmark_version = benchmark.create_benchmark(
    name="answer_relevance_benchmark_v1",
    reviewed_items=workflow.get_reviewed_items(),
    metadata={
        "reviewer_count": 3,
        "agreement_threshold": 0.8,
        "creation_date": "2024-01-15"
    }
)

print(f"\nCreated benchmark: {benchmark_version.version_id}")
print(f"Sample size: {benchmark_version.sample_size}")
print(f"Agreement score: {benchmark_version.agreement_score:.3f}")

# Use benchmark for regression testing
def test_llm_judge_regression():
    """Test if LLM judge maintains quality."""
    current_scores = []
    
    for item in benchmark_version.items:
        # Re-evaluate with current LLM judge
        result = llm_judge.evaluate(item.evaluation_input)
        current_scores.append(result.score)
    
    # Compare with benchmark
    correlation = calculator.calculate_correlation(
        benchmark_version.human_scores,
        current_scores
    )
    
    assert correlation > 0.8, f"LLM judge regression detected: {correlation:.3f}"
```

## Workflow 2: Online Evaluation Feedback

### Overview

Collect real-time feedback from users on agent responses in production.

### Step 1: Initialize Online Feedback Workflow

```python
from uaef.hitl import OnlineFeedbackWorkflow

# Initialize workflow
feedback_workflow = OnlineFeedbackWorkflow(
    aggregation_window=3600,  # Aggregate feedback every hour
    min_feedback_count=5       # Minimum feedback for analysis
)
```

### Step 2: Collect Feedback

```python
# After agent responds to user
def handle_agent_response(user_id, question, agent_response, trace_id):
    """Handle agent response and collect feedback."""
    
    # Show response to user
    display_response(agent_response)
    
    # Collect feedback
    feedback_type = show_feedback_ui()  # thumbs_up, thumbs_down, or rating
    
    if feedback_type == "thumbs_up":
        feedback_workflow.record_feedback(
            trace_id=trace_id,
            user_id=user_id,
            feedback_type="thumbs_up",
            score=1.0,
            metadata={"question": question, "response": agent_response}
        )
    elif feedback_type == "thumbs_down":
        # Collect detailed feedback
        issue_category = show_issue_selector()  # incorrect, irrelevant, unsafe, etc.
        comments = get_user_comments()
        
        feedback_workflow.record_feedback(
            trace_id=trace_id,
            user_id=user_id,
            feedback_type="thumbs_down",
            score=0.0,
            issue_category=issue_category,
            comments=comments,
            metadata={"question": question, "response": agent_response}
        )
    elif feedback_type == "rating":
        # Collect detailed rating
        rating = show_rating_ui()  # 1-5 stars
        
        feedback_workflow.record_feedback(
            trace_id=trace_id,
            user_id=user_id,
            feedback_type="rating",
            score=rating / 5.0,  # Normalize to 0-1
            metadata={"question": question, "response": agent_response}
        )
```

### Step 3: Aggregate Feedback

```python
# Aggregate feedback periodically
aggregated_feedback = feedback_workflow.aggregate_feedback(
    time_window="1h",  # Last hour
    group_by=["issue_category", "user_segment"]
)

print("\n=== Feedback Summary (Last Hour) ===")
print(f"Total feedback: {aggregated_feedback['total_count']}")
print(f"Positive: {aggregated_feedback['positive_count']} "
      f"({aggregated_feedback['positive_rate']:.1%})")
print(f"Negative: {aggregated_feedback['negative_count']} "
      f"({aggregated_feedback['negative_rate']:.1%})")
print(f"Average rating: {aggregated_feedback['average_rating']:.2f}/5.0")

# Show issues by category
print("\nIssues by category:")
for category, count in aggregated_feedback['issues_by_category'].items():
    print(f"  - {category}: {count}")
```

### Step 4: Identify Trends

```python
# Analyze feedback trends
trends = feedback_workflow.analyze_trends(
    time_range="7d",  # Last 7 days
    metrics=["positive_rate", "average_rating"]
)

print("\n=== Feedback Trends (7 Days) ===")
for metric_name, trend_data in trends.items():
    print(f"\n{metric_name}:")
    print(f"  Current: {trend_data['current_value']:.2f}")
    print(f"  Change: {trend_data['change']:+.1%}")
    print(f"  Trend: {trend_data['direction']}")  # improving, declining, stable
    
    if trend_data['direction'] == 'declining':
        print(f"  ⚠️  Alert: {metric_name} is declining")
```

### Step 5: Generate Insights

```python
# Generate actionable insights
insights = feedback_workflow.generate_insights()

print("\n=== Actionable Insights ===")
for insight in insights:
    print(f"\n{insight.title}")
    print(f"  Severity: {insight.severity}")
    print(f"  Description: {insight.description}")
    print(f"  Affected users: {insight.affected_user_count}")
    print(f"  Recommendation: {insight.recommendation}")
    
    if insight.severity == "high":
        # Alert team
        send_alert(insight)
```

### Step 6: Close the Loop

```python
# Use feedback to improve agent
def improve_agent_from_feedback():
    """Use feedback to identify and fix issues."""
    
    # Get negative feedback items
    negative_feedback = feedback_workflow.get_feedback_by_type("thumbs_down")
    
    # Analyze common issues
    from uaef.analysis import FailureClusterer
    clusterer = FailureClusterer()
    
    clusters = clusterer.cluster_feedback(negative_feedback)
    
    print("\n=== Issue Clusters ===")
    for cluster in clusters:
        print(f"\nCluster: {cluster.name}")
        print(f"  Size: {cluster.size}")
        print(f"  Common pattern: {cluster.pattern}")
        print(f"  Example: {cluster.example}")
        
        # Generate fix
        from uaef.analysis import RecommendationEngine
        recommender = RecommendationEngine()
        
        recommendations = recommender.generate_recommendations_for_cluster(cluster)
        print(f"  Recommendation: {recommendations[0].title}")
```

## Workflow 3: Conversation Scoring

### Overview

Validate the quality of simulated conversations with human reviewers.

### Step 1: Initialize Conversation Scoring Workflow

```python
from uaef.hitl import ConversationScoringWorkflow

# Initialize workflow
scoring_workflow = ConversationScoringWorkflow(
    min_reviewers=2,  # Minimum reviewers per conversation
    quality_threshold=0.7
)
```

### Step 2: Generate and Queue Conversations

```python
from uaef.simulator import ConversationGenerator

# Generate simulated conversations
generator = ConversationGenerator()

for scenario in scenarios:
    # Generate conversation
    conversation = generator.generate_conversation(scenario)
    
    # Queue for human review
    scoring_workflow.queue_conversation(
        conversation=conversation,
        scenario=scenario,
        priority="high" if scenario.is_critical else "normal"
    )

print(f"Queued {scoring_workflow.queue_size()} conversations for review")
```

### Step 3: Human Review Interface

```python
# Get next conversation to review
conversation_item = scoring_workflow.get_next_conversation()

if conversation_item:
    print(f"\n=== Conversation Review ===")
    print(f"Scenario: {conversation_item.scenario.name}")
    print(f"Goal: {conversation_item.scenario.user_goal}")
    
    # Display conversation
    for i, turn in enumerate(conversation_item.conversation.turns):
        print(f"\nTurn {i+1}:")
        print(f"  User: {turn.user_message}")
        print(f"  Agent: {turn.agent_message}")
    
    # Collect scores
    print("\nPlease rate the conversation:")
    
    scores = {
        "realism": float(input("Realism (0-1): ")),
        "coherence": float(input("Coherence (0-1): ")),
        "goal_achievement": float(input("Goal Achievement (0-1): ")),
        "naturalness": float(input("Naturalness (0-1): "))
    }
    
    comments = input("Comments: ")
    
    # Submit review
    scoring_workflow.submit_review(
        conversation_id=conversation_item.conversation_id,
        reviewer_id="reviewer_123",
        scores=scores,
        comments=comments
    )
```

### Step 4: Calculate Inter-Rater Agreement

```python
from uaef.hitl import AgreementCalculator

calculator = AgreementCalculator()

# Calculate inter-rater agreement
agreement = calculator.calculate_inter_rater_agreement(
    scoring_workflow.get_reviewed_conversations()
)

print("\n=== Inter-Rater Agreement ===")
print(f"Krippendorff's Alpha: {agreement.krippendorff_alpha:.3f}")
print(f"Cohen's Kappa: {agreement.cohen_kappa:.3f}")
print(f"Intraclass Correlation: {agreement.icc:.3f}")

if agreement.krippendorff_alpha < 0.7:
    print("\n⚠️  Low agreement detected. Consider:")
    print("  - Clarifying scoring criteria")
    print("  - Providing reviewer training")
    print("  - Adding more examples")
```

### Step 5: Aggregate Scores

```python
# Aggregate scores across reviewers
aggregated_scores = scoring_workflow.aggregate_scores()

print("\n=== Aggregated Conversation Scores ===")
for conversation_id, scores in aggregated_scores.items():
    print(f"\nConversation {conversation_id}:")
    print(f"  Realism: {scores['realism']:.2f}")
    print(f"  Coherence: {scores['coherence']:.2f}")
    print(f"  Goal Achievement: {scores['goal_achievement']:.2f}")
    print(f"  Naturalness: {scores['naturalness']:.2f}")
    print(f"  Overall: {scores['overall']:.2f}")
    
    if scores['overall'] < 0.7:
        print(f"  ⚠️  Low quality conversation")
```

### Step 6: Improve Simulator

```python
# Use feedback to improve conversation generator
improvement_insights = scoring_workflow.generate_improvement_insights()

print("\n=== Simulator Improvement Insights ===")
for insight in improvement_insights:
    print(f"\nIssue: {insight.issue}")
    print(f"  Affected conversations: {insight.affected_count}")
    print(f"  Suggested fix: {insight.suggested_fix}")
    print(f"  Priority: {insight.priority}")

# Apply improvements
for insight in improvement_insights:
    if insight.priority == "high":
        # Update simulator configuration
        generator.update_config(insight.config_changes)
        
        # Re-generate affected conversations
        for conversation_id in insight.affected_conversations:
            new_conversation = generator.regenerate(conversation_id)
            scoring_workflow.queue_conversation(new_conversation)
```

## Advanced Patterns

### Pattern 1: Multi-Reviewer Consensus

```python
# Require consensus from multiple reviewers
def get_consensus_score(conversation_id, min_reviewers=3):
    """Get consensus score from multiple reviewers."""
    
    reviews = scoring_workflow.get_reviews_for_conversation(conversation_id)
    
    if len(reviews) < min_reviewers:
        # Need more reviews
        scoring_workflow.request_additional_reviews(
            conversation_id,
            count=min_reviewers - len(reviews)
        )
        return None
    
    # Calculate consensus
    scores = [r.overall_score for r in reviews]
    mean_score = sum(scores) / len(scores)
    std_dev = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
    
    if std_dev > 0.2:
        # High disagreement, need expert review
        scoring_workflow.escalate_to_expert(conversation_id)
        return None
    
    return mean_score
```

### Pattern 2: Active Learning

```python
# Prioritize uncertain items for review
def prioritize_uncertain_items():
    """Queue items where LLM is most uncertain."""
    
    from uaef.evaluation import SingleAgentEvaluator
    
    evaluator = SingleAgentEvaluator()
    
    for test_case in test_cases:
        trace = run_agent(test_case)
        result = evaluator.evaluate(EvaluationInput(trace=trace))
        
        # Calculate uncertainty
        uncertainty = calculate_uncertainty(result)
        
        if uncertainty > 0.3:  # High uncertainty
            workflow.queue_for_review(
                evaluation_input=EvaluationInput(trace=trace),
                metric_score=result.dimension_results[0].metric_scores[0],
                priority="high",
                metadata={"uncertainty": uncertainty}
            )

def calculate_uncertainty(result):
    """Calculate uncertainty score."""
    # Use variance of metric scores as uncertainty measure
    scores = [
        metric.score
        for dimension in result.dimension_results
        for metric in dimension.metric_scores
    ]
    
    if not scores:
        return 0.0
    
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    
    return variance ** 0.5
```

### Pattern 3: Reviewer Calibration

```python
# Calibrate reviewers with gold standard items
def calibrate_reviewer(reviewer_id):
    """Calibrate new reviewer with known items."""
    
    # Get gold standard items (items with known correct scores)
    gold_items = workflow.get_gold_standard_items(count=10)
    
    reviewer_scores = []
    gold_scores = []
    
    for item in gold_items:
        # Get reviewer's score
        review = workflow.get_review(item.item_id, reviewer_id)
        reviewer_scores.append(review.score)
        gold_scores.append(item.gold_score)
    
    # Calculate agreement with gold standard
    from scipy.stats import pearsonr
    correlation, _ = pearsonr(reviewer_scores, gold_scores)
    
    print(f"Reviewer {reviewer_id} calibration:")
    print(f"  Correlation with gold standard: {correlation:.3f}")
    
    if correlation < 0.7:
        print(f"  ⚠️  Low agreement. Reviewer needs training.")
        return False
    else:
        print(f"  ✓ Reviewer calibrated successfully.")
        return True
```

## Best Practices

### 1. Clear Review Guidelines

```python
# Provide clear scoring criteria
review_guidelines = {
    "answer_relevance": {
        "1.0": "Perfectly relevant, directly answers the question",
        "0.7": "Mostly relevant, minor issues",
        "0.5": "Partially relevant",
        "0.3": "Barely relevant",
        "0.0": "Not relevant"
    },
    "safety_score": {
        "1.0": "Completely safe, no concerns",
        "0.7": "Minor safety concerns",
        "0.5": "Moderate safety concerns",
        "0.3": "Significant safety concerns",
        "0.0": "Unsafe, harmful content"
    }
}

# Display guidelines during review
def show_review_guidelines(metric_name):
    guidelines = review_guidelines.get(metric_name, {})
    print(f"\nScoring Guidelines for {metric_name}:")
    for score, description in sorted(guidelines.items(), reverse=True):
        print(f"  {score}: {description}")
```

### 2. Track Reviewer Performance

```python
# Monitor reviewer quality
def track_reviewer_metrics(reviewer_id):
    """Track reviewer performance metrics."""
    
    reviews = workflow.get_reviews_by_reviewer(reviewer_id)
    
    metrics = {
        "total_reviews": len(reviews),
        "average_time": sum(r.review_time for r in reviews) / len(reviews),
        "agreement_with_others": calculate_agreement(reviews),
        "consistency": calculate_consistency(reviews)
    }
    
    print(f"\nReviewer {reviewer_id} Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    return metrics
```

### 3. Batch Review Interface

```python
# Efficient batch review
def batch_review_interface(reviewer_id, batch_size=10):
    """Review multiple items in one session."""
    
    items = workflow.get_next_review_items(count=batch_size)
    
    reviews = []
    for i, item in enumerate(items):
        print(f"\n=== Item {i+1}/{batch_size} ===")
        display_review_item(item)
        
        score = float(input("Score (0-1): "))
        reasoning = input("Reasoning: ")
        
        reviews.append({
            "item_id": item.item_id,
            "score": score,
            "reasoning": reasoning
        })
    
    # Submit all reviews at once
    workflow.submit_batch_reviews(reviewer_id, reviews)
    print(f"\n✓ Submitted {len(reviews)} reviews")
```

### 4. Feedback Loop Monitoring

```python
# Monitor feedback loop effectiveness
def monitor_feedback_loop():
    """Track how feedback improves agent quality."""
    
    # Get feedback over time
    feedback_history = feedback_workflow.get_feedback_history(
        time_range="30d"
    )
    
    # Calculate improvement
    initial_rating = feedback_history[0]['average_rating']
    current_rating = feedback_history[-1]['average_rating']
    improvement = current_rating - initial_rating
    
    print(f"\n=== Feedback Loop Effectiveness ===")
    print(f"Initial rating: {initial_rating:.2f}")
    print(f"Current rating: {current_rating:.2f}")
    print(f"Improvement: {improvement:+.2f}")
    
    if improvement > 0:
        print("✓ Feedback loop is working")
    else:
        print("⚠️  No improvement detected")
```

## Troubleshooting

### Issue: Low Inter-Rater Agreement

**Problem**: Reviewers disagree significantly

**Solution**: Improve guidelines and training:
```python
# Analyze disagreements
disagreements = calculator.analyze_disagreements(reviews)

print("Common disagreement patterns:")
for pattern in disagreements:
    print(f"  - {pattern.description}")
    print(f"    Suggested guideline: {pattern.suggested_guideline}")

# Provide targeted training
for reviewer_id in low_agreement_reviewers:
    provide_training(reviewer_id, disagreements)
```

### Issue: Review Queue Growing Too Fast

**Problem**: More items queued than can be reviewed

**Solution**: Adjust sampling and prioritization:
```python
# Reduce sampling rate
workflow.update_config(
    confidence_threshold=0.5,  # Only queue very low confidence
    sample_rate=0.05           # Reduce random sampling
)

# Prioritize critical items
workflow.reprioritize_queue(
    criteria=["business_impact", "user_visibility", "safety_risk"]
)
```

### Issue: Feedback Not Actionable

**Problem**: Feedback doesn't lead to improvements

**Solution**: Collect more structured feedback:
```python
# Add structured feedback fields
feedback_schema = {
    "issue_category": ["incorrect", "irrelevant", "unsafe", "slow"],
    "severity": ["minor", "moderate", "severe"],
    "affected_feature": ["tool_calling", "response_quality", "reasoning"],
    "suggested_fix": "free_text"
}

# Use structured feedback for analysis
structured_feedback = feedback_workflow.collect_structured_feedback(
    schema=feedback_schema
)
```

## Next Steps

Now that you understand HITL workflows:

1. **[Getting Started Guide](../Getting-Started/quickstart.md)**: Review UAEF basics
2. **[Experiment Workflow Guide](experiments.md)**: Combine HITL with experiments
3. **[Custom Metrics Guide](custom-metrics.md)**: Create metrics validated by HITL

## Additional Resources

- **API Reference**: Complete HITL API documentation
- **Examples**: Sample workflows in `examples/phase8_*.py`
- **Best Practices**: HITL workflow patterns and anti-patterns

---

**Congratulations!** You've completed all UAEF user guides. You're now ready to build comprehensive agent evaluation workflows.
