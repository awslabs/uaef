# Metrics Implementation Issues & Improvements
 
**Date**: 2026-04-15  
**Status**: Proposed

---

## Executive Summary: Issues & Proposed Solutions

| # | Metric | Issue | Impact | Proposed Solution | Priority |
|---|--------|-------|--------|-------------------|----------|
| **1** | [**Tool Selection Accuracy**](#tool-selection-accuracy---critical-issue) | Swapped args → 100% (bag-of-values) | ❌ False positives | Per-call matching (key-value pairs) | 🔴 High |
| | | Separated eval + fixed weights | ⚠️ Inflexible | Holistic per-call scoring | 🔴 High |
| **2** | [**Parameter Quality**](#parameter-quality-metric---same-issue) | Same `_args_acc()` function issue | ❌ False positives | Same per-call matching fix | 🔴 High |
| **3** | [**Edge Case Handling**](#edge-case-handling---score-inflationdeflation-issue) | Returns 1.0/0.0 for non-applicable (64 cases) | ⚠️ Score inflation/deflation | Return None, filter in aggregation | 🔴 High |
| **4** | [**Token Efficiency**](#tokencost-efficiency-metrics---circular-dependency-issue) | quality_score never set (always 0.5) | ❌ Meaningless scores | Two-phase: calc quality first | 🟡 Medium |
| | | Flat < 1000, harsh above | ⚠️ Poor scaling | Threshold + penalty modes (√/linear) | 🟡 Medium |
| **5** | [**Cost Efficiency**](#cost-efficiency---same-issues--critical-bug) | Same quality_score issue | ❌ Meaningless scores | Two-phase evaluation | 🟡 Medium |
| | | **`max(0.01, ...)` should be `max(1.0, ...)`** | ❌ **Wrong threshold** | **Fix bug** | 🔴 **Critical** |
| **6** | [**Context Retention**](#context-retention---false-positive-for-non-retention-conversations) | High scores for independent questions | ❌ False positives | Return None if no retention needed | 🟡 Medium |
| **7** | [**Conversation Completeness**](#conversation-completeness---make-success-criteria-optional) | Requires explicit success criteria | ⚠️ Inflexible | Make optional, extract intentions | 🟡 Medium |
| **8** | [**Turn Efficiency**](#turn-efficiency---incorporate-completeness) | Ignores goal achievement | ⚠️ Incomplete metric | Include completeness in formula | 🟡 Medium |
| **9** | [**Agent Utilization**](#agent-utilization---single-vs-batch-evaluation) | Single-query limited meaning | ℹ️ Context-dependent | Add batch calculation, document | 🟢 Low |
| **10** | [**Workflow Completion**](#workflow-completion---binary-failure-handling) | Binary failure (0.0 for any failure) | ❌ No partial credit | Evaluate partial + apply penalty | 🟡 Medium |
| **11** | [**Coordination Efficiency**](#coordination-efficiency---hardcoded-ideal-ratio) | Hardcoded `ideal_ratio = 0.1` | ⚠️ One-size-fits-all | Make configurable per workflow | 🟢 Low |
| **12** | [**Missing Metrics**](#missing-metrics---use-deepeval-integration) | No plan quality/adherence/error recovery | ⚠️ Coverage gaps | Use DeepEval + add error recovery | 🟡 Medium |

**Legend**: 🔴 High Priority | 🟡 Medium | 🟢 Low Priority

**Quick Stats:**
- Total issues: 12 categories, 20+ specific problems
- High priority: 4 (critical bugs, false positives)
- Quick wins: Cost efficiency bug fix, edge case documentation

---

## Tool Selection Accuracy - Critical Issue

### Problem

Current `ToolSelectionAccuracyMetric` uses **bag-of-values** comparison that **cannot detect swapped arguments**.

### Concrete Example

```python
Expected: subtract(a=100, b=25)  # Result: 75
Called:   subtract(a=25, b=100)  # Result: -75 (WRONG!)

Current Score: 100% ✅ (FALSE POSITIVE)
Should Be:     0%   ❌ (Arguments swapped)
```

**Why?** Current implementation:
```python
# Groups by tool name, uses Counter (order-independent)
Expected args: ['100', '25']
Called args:   ['25', '100']

Counter(['100', '25']) == Counter(['25', '100'])  # TRUE
Score: 100%  # WRONG!
```

### More Examples Where Current Implementation Fails

| Example | Expected | Called | Current Score | Should Be | Impact |
|---------|----------|--------|---------------|-----------|--------|
| **Math** | `subtract(100, 25)` | `subtract(25, 100)` | 100% | 0% | Wrong result! |
| **Coordinates** | `set_pixel(x=10, y=20)` | `set_pixel(x=20, y=10)` | 100% | 33% | Wrong location! |
| **String ops** | `replace(text, "old", "new")` | `replace(text, "new", "old")` | 100% | 33% | Wrong replacement! |
| **Extra params** | `call(a=1)` | `call(a=1, b=2, c=3, d=4)` | 100% | 25%* | Too lenient |

*If extras should be penalized

### Root Cause #1: Bag-of-Values Semantics

```python
# Current: pools all args by tool name (loses per-call info)
gt_args_dict = {}
for i in range(len(gt_tools)):
    if gt_tools[i] in gt_args_dict:
        gt_args_dict[gt_tools[i]].extend(gt_args[i])  # Pooling!
```

### Root Cause #2: Separated Evaluation with Fixed Weights

**Current Formula:**
```python
score = 1.0 - (0.4 × incorrect_pct + 0.4 × missed_pct + 0.2 × (1 - args_acc))
```

**Problems:**
1. **Separates tool name matching from parameter matching** - evaluates them independently
2. **Hardcoded weights** (40% incorrect, 40% missed, 20% args) - no flexibility
3. **Combines orthogonal errors** - can get high score even with completely wrong parameters

**Example of the Issue:**
```python
Expected: subtract(a=100, b=25)
Called:   subtract(a=25, b=100)

incorrect_pct = 0.0    # ✓ Tool name correct
missed_pct = 0.0       # ✓ No tools missed  
args_acc = 1.0         # ✓ Arguments present (bag semantics)

score = 1.0 - (0.4×0 + 0.4×0 + 0.2×0) = 1.0  # FALSE POSITIVE!
```

The formula gives perfect score because:
- Tool **name** is correct (subtract ✓)
- All argument **values** are present ([100, 25] ✓)
- But **binding** is wrong (a=25 should be a=100)

---

## Proposed Solution: DeepEval's Holistic Approach

### Key Idea: Count Correct Tool Calls (Not Separate Components)

Instead of separating tool names and parameters with fixed weights, **evaluate each tool call as a complete unit**.

**DeepEval's Approach:**
```python
def calculate_tool_correctness(expected_calls, actual_calls):
    """
    For each expected call, find best matching actual call.
    Score = average match quality across expected calls.
    """
    total_score = 0.0
    matched = set()
    
    for expected in expected_calls:
        best_score = 0.0
        
        for idx, actual in enumerate(actual_calls):
            if idx in matched:
                continue
            
            # Both name AND params must match for score
            if expected.name == actual.name:
                # Compare params as key-value pairs (order matters!)
                param_score = compare_params(expected.arguments, actual.arguments)
                
                if param_score > best_score:
                    best_score = param_score
                    best_idx = idx
        
        if best_score > 0:
            total_score += best_score
            matched.add(best_idx)
    
    return total_score / len(expected_calls)

def compare_params(expected, actual):
    """Key-value pair comparison."""
    matched_keys = sum(1 for k, v in expected.items() 
                       if k in actual and actual[k] == v)
    total_keys = len(set(expected.keys()) | set(actual.keys()))
    return matched_keys / total_keys if total_keys > 0 else 1.0
```

### Why This Is Better

| Aspect | Current (UAEF) | Proposed (DeepEval-style) |
|--------|----------------|---------------------------|
| **Evaluation** | Tool names and params **separately** | Tool calls **holistically** |
| **Weights** | **Fixed** (40%, 40%, 20%) | **Natural** (per-call quality) |
| **Parameter matching** | Bag-of-values (order ignored) | Key-value pairs (order respected) |
| **Swapped args** | ❌ Not detected (100%) | ✅ Detected (0%) |
| **Per-call tracking** | ❌ No | ✅ Yes |

### Results with Fix

| Example | Expected | Called | Current | Proposed | Impact |
|---------|----------|--------|---------|----------|--------|
| **Swapped args** | `subtract(a=100, b=25)` | `subtract(a=25, b=100)` | 100% ❌ | 0% ✅ | Detects error |
| **Coordinates** | `set_pixel(x=10, y=20, c='red')` | `set_pixel(x=20, y=10, c='red')` | 100% ❌ | 33% ✅ | Detects swap |
| **Perfect match** | `search(q='Paris')` | `search(q='Paris')` | 100% ✅ | 100% ✅ | Same |
| **Missing param** | `call(a=1, b=2)` | `call(a=1)` | 50% ✅ | 50% ✅ | Same |

---

## Implementation Plan

### Recommended: Replace Formula with DeepEval-Style Scoring

**Current:**
```python
# Separate evaluation with fixed weights
incorrect_pct = _incorrect_tool_pct(gt_tools, called_tools)  
missed_pct = _missed_tool_pct(gt_tools, called_tools)
args_acc = _args_acc(gt_tools, gt_args, called_tools, called_tools_args)

score = 1.0 - (0.4 × incorrect_pct + 0.4 × missed_pct + 0.2 × (1 - args_acc))
```

**Proposed:**
```python
# Holistic evaluation: count correct tool calls
score = _calculate_correct_tool_calls(expected_tool_calls, actual_tool_calls)

# Each tool call is correct if:
# 1. Tool name matches
# 2. Parameters match (key-value pairs)
```

### Migration Options

| Option | Approach | Breaking? | Effort | Recommendation |
|--------|----------|-----------|--------|----------------|
| **A: Add Mode** | `mode="legacy"` vs `"strict"` | No (with deprecation) | Medium | ⭐ **Best** |
| **B: Fix in Place** | Replace implementation | Yes | Low | ⚠️ Risky |
| **C: New Metric** | `ToolCallCorrectnessMetric` | No | High | 🔄 Fallback |

**Recommendation**: **Option A** - Add mode parameter, gradual migration over 6 months

---

## Parameter Quality Metric - Same Issue

### Current Implementation

`ParameterQualityMetric` (lines 389-471 in `tool_calling.py`) **uses the exact same `_args_acc()` function**:

```python
class ParameterQualityMetric(BaseMetric):
    """Evaluates the quality and correctness of tool arguments."""
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        # ... validation ...
        
        # Uses same problematic function!
        args_accuracy = _args_acc(gt_tools, gt_args, called_tools, called_tools_args)
        
        return MetricScore(
            metric_name=self.get_name(),
            score=args_accuracy,  # Direct score from _args_acc
            reasoning=f"Parameter accuracy: {args_accuracy:.1%}",
            metadata={"args_accuracy": args_accuracy}
        )
```

### Same Problems

| Issue | Impact on ParameterQualityMetric |
|-------|----------------------------------|
| Bag-of-values | ❌ Cannot detect swapped arguments |
| Tool name grouping | ❌ Cannot track per-call parameters |
| Edge case bug | ❌ Returns 1.0 when no expected tools |

**Example:**
```python
Expected: divide(dividend=100, divisor=5)  # Result: 20
Called:   divide(dividend=5, divisor=100)  # Result: 0.05 (WRONG!)

Current ParameterQualityMetric Score: 100% ❌
Should Be: 0%
```

### Proposed Fix

**Apply the same DeepEval-style approach:**

```python
class ParameterQualityMetric(BaseMetric):
    def __init__(self, mode: str = "legacy", allow_extra_params: bool = True):
        self.mode = mode
        self.allow_extra_params = allow_extra_params
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        # ... validation ...
        
        if self.mode == "strict":
            # NEW: Per-call parameter matching
            score = _calculate_parameter_quality(
                gt_tool_calls,
                actual_tool_calls,
                allow_extra=self.allow_extra_params
            )
        else:
            # Legacy: bag-of-values
            score = _args_acc(gt_tools, gt_args, called_tools, called_tools_args)
        
        return MetricScore(
            metric_name=self.get_name(),
            score=score,
            reasoning=f"[{self.mode.upper()}] Parameter accuracy: {score:.1%}",
            metadata={"mode": self.mode, "args_accuracy": score}
        )

def _calculate_parameter_quality(expected_calls, actual_calls, allow_extra=True):
    """
    Calculate parameter quality with per-call matching.
    Same algorithm as tool correctness, but focuses only on parameters.
    """
    total_score = 0.0
    matched = set()
    
    for expected in expected_calls:
        best_score = 0.0
        
        for idx, actual in enumerate(actual_calls):
            if idx in matched or expected.name != actual.name:
                continue
            
            # Only evaluate parameter quality
            param_score = _compare_params(
                expected.arguments,
                actual.arguments,
                allow_extra=allow_extra
            )
            
            if param_score > best_score:
                best_score = param_score
                best_idx = idx
        
        if best_score > 0:
            total_score += best_score
            matched.add(best_idx)
    
    return total_score / len(expected_calls) if expected_calls else 1.0
```

**Benefits:**
- ✅ Detects swapped arguments
- ✅ Consistent with `ToolSelectionAccuracyMetric` improvement
- ✅ Same migration path (modes + deprecation)

---

## Summary: Two Metrics Need Updates

| Metric | Current Issue | Proposed Fix | Priority |
|--------|--------------|--------------|----------|
| `ToolSelectionAccuracyMetric` | Separated evaluation + bag semantics + fixed weights | Holistic per-call matching | 🔴 High |
| `ParameterQualityMetric` | Uses same `_args_acc()` function | Use same per-call matching | 🔴 High |

**Both should be updated together** to maintain consistency.

---

## Edge Case Handling - Score Inflation/Deflation Issue

### Problem: Non-Applicable Metrics Return Scores

Many metrics return `1.0` or `0.0` when they **cannot be evaluated**, artificially affecting aggregated scores.

### Examples of Score Inflation (Returns 1.0)

#### 1. MCP Compliance - No Tool Calls
```python
# src/uaef/metrics/tool_calling.py, line 505
if len(tool_calls) == 0:
    return MetricScore(
        metric_name=self.get_name(),
        score=1.0,  # ❌ Inflates score
        reasoning="No tool calls to evaluate for MCP compliance",
        metadata={"tool_count": 0}
    )
```

**Issue**: Agent gets "perfect MCP compliance" without making any tool calls. Inflates overall dimension score.

#### 2. Hallucination Score - No Context
```python
# src/uaef/metrics/response_quality.py, line 467
if not context or len(context) == 0:
    return MetricScore(
        metric_name=self.get_name(),
        score=1.0,  # ❌ Inflates score
        reasoning="No context provided, cannot detect hallucinations",
        metadata={"warning": "no_context"}
    )
```

#### 3. Context Retention - Too Few Turns
```python
# src/uaef/metrics/multi_turn.py, line 51
if len(trace.messages) < 4:
    return MetricScore(
        metric_name=self.get_name(),
        score=1.0,  # ❌ Inflates score
        reasoning="Not enough turns to evaluate context retention",
        metadata={"message_count": len(trace.messages)}
    )
```

### Examples of Score Deflation (Returns 0.0)

#### 1. Answer Relevance - No Question Found
```python
# src/uaef/metrics/response_quality.py, line 64
if not question:
    return MetricScore(
        metric_name=self.get_name(),
        score=0.0,  # ❌ Deflates score
        reasoning="No user question found in trace",
        metadata={"error": "missing_question"}
    )
```

**Issue**: Data issue (malformed trace) penalizes agent with 0% relevance score.

#### 2. Answer Relevance - Bedrock API Error
```python
# src/uaef/metrics/response_quality.py, line 104
except Exception as e:
    return MetricScore(
        metric_name=self.get_name(),
        score=0.0,  # ❌ Deflates score
        reasoning=f"Error calling LLM judge: {str(e)}",
        metadata={"error": str(e)}
    )
```

**Issue**: Infrastructure failure (AWS outage) gives agent 0% score.

### Impact on Aggregated Scores

**Example Evaluation:**
```python
# Agent with 3 tool calls, all correct
Metrics:
  - tool_selection_accuracy: 0.95 (legitimate)
  - mcp_compliance: 1.0 (inflated - just 3 tool calls evaluated)
  - parameter_quality: 0.90 (legitimate)

Average: (0.95 + 1.0 + 0.90) / 3 = 0.95  # Artificially high
```

**Example with Errors:**
```python
# Agent response is good, but Bedrock API is down
Metrics:
  - answer_relevance: 0.0 (deflated - API error, not agent issue)
  - completeness: 0.0 (deflated - API error)
  - accuracy: 0.0 (deflated - API error)

Average: 0.0  # Artificially low due to infrastructure
```

---

## Proposed Solution: Return None for Non-Applicable Cases

### Update MetricScore Model

```python
# src/uaef/models/metric_score.py
class MetricScore(BaseModel):
    metric_name: str
    score: Optional[float]  # Allow None
    reasoning: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = {}
    applicable: bool = True  # NEW: indicates if metric could be evaluated
```

### Update Metrics to Return None + Warning

**Unified pattern with edge case type tracking:**

```python
# MCP Compliance - Not Applicable
if len(tool_calls) == 0:
    return MetricScore(
        metric_name=self.get_name(),
        score=None,
        applicable=False,
        reasoning="Cannot evaluate: no tool calls in trace",
        metadata={
            "warning": "edge_case",
            "edge_case_type": "not_applicable",
            "tool_count": 0
        }
    )

# Answer Relevance - Missing Data (no redundant ground_truth check)
question = self._extract_question(trace)
if not question:
    return MetricScore(
        metric_name=self.get_name(),
        score=None,
        applicable=False,
        reasoning="Cannot evaluate: no user question in trace",
        metadata={
            "warning": "edge_case",
            "edge_case_type": "missing_data"
        }
    )

# Answer Relevance - API Error
except boto3.exceptions.ClientError as e:
    return MetricScore(
        metric_name=self.get_name(),
        score=None,
        applicable=False,
        reasoning="Cannot evaluate: Bedrock API error",
        metadata={
            "warning": "edge_case",
            "edge_case_type": "api_error",
            "error_details": str(e)
        }
    )
```

### Update Aggregation Logic

```python
# src/uaef/evaluation/single_agent.py
def _aggregate_dimension_scores(self, metric_scores: List[MetricScore]) -> float:
    """Aggregate scores, filtering out non-applicable metrics."""
    
    # Filter out None scores
    applicable_scores = [
        s.score for s in metric_scores 
        if s.score is not None and s.applicable
    ]
    
    if not applicable_scores:
        # No applicable metrics - return None for dimension too
        return None
    
    return sum(applicable_scores) / len(applicable_scores)

def _calculate_overall_score(self, dimension_results: Dict[str, DimensionResult]) -> float:
    """Calculate overall score, filtering out non-applicable dimensions."""
    
    applicable_dimensions = [
        (name, result) for name, result in dimension_results.items()
        if result.score is not None
    ]
    
    if not applicable_dimensions:
        return None  # Cannot calculate overall score
    
    # Weight by dimension weights
    weighted_sum = sum(
        result.score * self.dimension_weights.get(name, 1.0)
        for name, result in applicable_dimensions
    )
    
    total_weight = sum(
        self.dimension_weights.get(name, 1.0)
        for name, result in applicable_dimensions
    )
    
    return weighted_sum / total_weight
```

---

## When Should Metrics Return None?

| Situation | Current | Proposed | Reason |
|-----------|---------|----------|--------|
| **No data to evaluate** | 1.0 | `None` | Nothing to measure |
| **Data missing** (malformed trace) | 0.0 | `None` | Not agent's fault |
| **API/Infrastructure error** | 0.0 | `None` | Not agent's fault |
| **Metric not applicable** | 1.0 | `None` | Cannot evaluate |
| **Agent actually failed** | 0.0 | 0.0 | Legitimate failure |

**Rule of Thumb**: If the metric **cannot be evaluated** (not the agent's fault), return `None + warning`. If the agent **performed poorly**, return low score.

**Important**: Simplify redundant ground_truth checks:
```python
# Instead of double-checking:
if not evaluation_input.ground_truth:
    raise ValueError("Ground truth required")
expected_output = evaluation_input.ground_truth.expected_output
if not expected_output:
    return MetricScore(score=1.0, ...)

# Just check what's needed:
expected_output = (evaluation_input.ground_truth.expected_output 
                   if evaluation_input.ground_truth else None)
if not expected_output:
    return MetricScore(
        score=None,
        applicable=False,
        metadata={"warning": "edge_case", "edge_case_type": "missing_data"}
    )
```

This applies to: completeness, accuracy, conversation_completeness, workflow_completion.

---

## Impact Analysis

### Before (Current)

```python
# Evaluation with 10 metrics
Scores: [0.9, 0.8, 1.0, 1.0, 1.0, 0.0, 0.0, 0.85, 0.75, 0.9]
         # Real scores: [0.9, 0.8, 0.85, 0.75, 0.9]
         # Inflated: [1.0, 1.0, 1.0] (no data to evaluate)
         # Deflated: [0.0, 0.0] (API errors)

Average: 0.72  # Misleading!
```

### After (Proposed)

```python
# Evaluation with 10 metrics
Scores: [0.9, 0.8, None, None, None, None, None, 0.85, 0.75, 0.9]
         # Real scores: [0.9, 0.8, 0.85, 0.75, 0.9]
         # Filtered out: [None, None, None, None, None]

Average: 0.84  # Accurate! (only applicable metrics)
```

**Result**: More accurate representation of agent quality.

---

## Common Non-Applicable Cases

| Metric | Returns None When... |
|--------|---------------------|
| `mcp_compliance` | No tool calls made |
| `tool_sequence_correctness` | No tool calls to sequence |
| `parameter_quality` | No tool calls to check |
| `answer_relevance` | No question or API error |
| `hallucination_score` | No context provided or API error |
| `completeness` | No expected output or API error |
| `accuracy` | No question/reference or API error |
| `safety_score` | No response or API error |
| `bias_score` | No response or API error |
| `toxicity_score` | No response or API error |
| `latency_score` | No latency metadata |
| `token_efficiency` | No token metadata |
| `cost_efficiency` | No cost metadata |
| `throughput` | No throughput metadata |
| `context_retention` | < 2 turns |
| `coherence` | < 2 turns |
| `conversation_completeness` | < 1 turn or no success criteria |
| `agent_utilization` | Not multi-agent trace |
| `delegation_quality` | Not multi-agent trace or no events |
| `workflow_completion` | Not multi-agent trace |
| `coordination_efficiency` | Not multi-agent trace |
| All reasoning metrics | No assistant messages |

---

## Implementation Changes Required

### 1. Update MetricScore Model
```python
# src/uaef/models/metric_score.py
score: Optional[float] = None  # Allow None
applicable: bool = True
```

### 2. Update All Metrics (~25 files)
- Change edge case returns to `score=None, applicable=False` with warning metadata
- Add `edge_case_type` to metadata: "missing_data", "not_applicable", "api_error"
- **Simplify redundant checks**: Remove double-checking of `ground_truth` and `expected_output`
- Distinguish between "cannot evaluate" (None) vs "evaluated poorly" (low score)

### 3. Update Aggregation Logic
- Filter `None` scores before averaging
- Handle case where all metrics return `None`

### 4. Update Tests
- Test that None scores are filtered correctly
- Test aggregation with mixed None/numeric scores

**Effort**: 1-2 weeks  
**Files affected**: ~30 (all metric files + aggregation logic)

---

## How DeepEval Handles Edge Cases

### DeepEval's Approach: Fail Fast with Exceptions

DeepEval **does NOT return default scores (1.0 or 0.0)**. Instead, it **raises exceptions** when required data is missing:

```python
# DeepEval validation (check_llm_test_case_params)
missing_params = []
for param in required_params:
    if getattr(test_case, param.value) is None:
        missing_params.append(f"'{param.value}'")

if missing_params:
    raise MissingTestCaseParamsError(
        f"{missing_params} cannot be None for the '{metric.__name__}' metric"
    )
```

**Examples:**

| Metric | Required Params | Missing Data Behavior |
|--------|----------------|----------------------|
| `ToolCorrectnessMetric` | `[INPUT, TOOLS_CALLED, EXPECTED_TOOLS]` | ❌ Raises exception |
| `HallucinationMetric` | `[INPUT, ACTUAL_OUTPUT, CONTEXT]` | ❌ Raises exception (no 1.0!) |
| `AnswerRelevancyMetric` | `[INPUT, ACTUAL_OUTPUT]` | ❌ Raises exception (no 0.0!) |

### Empty vs None: DeepEval's Distinction

DeepEval differentiates:
- **None/Missing**: Raises exception immediately (cannot evaluate)
- **Empty list `[]`**: Evaluates within metric logic

```python
# Tool Correctness with empty lists
if not expected_tools and not tools_called:
    return 1.0  # Both empty = correct behavior
elif not expected_tools:
    return 0.0  # No expected, but called some = wrong
else:
    return score / len(expected_tools)  # Normal calculation
```

**Key Insight**: Empty data is **semantically meaningful** (agent correctly did nothing), so it gets evaluated. Missing/None data is a **validation error** (cannot run metric).

---

## Recommended Approach for UAEF: Return None + Warning ⭐

### Why None + Warning is Better

**Advantages:**
- ✅ **Clear semantics**: None = "not applicable", 0.0 = "agent failed"
- ✅ **Pipeline doesn't break**: No exceptions thrown
- ✅ **Trackable**: Warning metadata with edge case types
- ✅ **Flexible**: Can still get partial results (some metrics applicable, some not)
- ✅ **Accurate aggregation**: Only count metrics that actually evaluated the agent
- ✅ **Better debugging**: Can analyze edge case patterns
- ✅ **Simpler code**: Eliminates redundant ground_truth checks

**Comparison:**

| Approach | When Not Applicable | Pros | Cons |
|----------|-------------------|------|------|
| **Exception** (DeepEval) | Raises error | Clear failure | Breaks pipeline, rigid |
| **None + Warning** (Recommended) | Returns None | No breaks, trackable | Requires aggregation logic |
| **Default score** (Current) | Returns 1.0 or 0.0 | Simple | Misleading, inaccurate |

### Unified Implementation Strategy

**All edge cases return None with warning metadata:**

```python
class MetricBase:
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        # Simplify redundant checks
        expected_output = (evaluation_input.ground_truth.expected_output 
                          if evaluation_input.ground_truth else None)
        # 1. MISSING DATA → None + warning
        if not expected_output:
            return MetricScore(
                score=None,
                applicable=False,
                reasoning="Cannot evaluate: no expected output",
                metadata={
                    "warning": "edge_case",
                    "edge_case_type": "missing_data"
                }
            )
        
        # 2. NOT APPLICABLE → None + warning
        if len(tool_calls) == 0:
            return MetricScore(
                score=None,
                applicable=False,
                reasoning="Cannot evaluate: no tool calls",
                metadata={
                    "warning": "edge_case",
                    "edge_case_type": "not_applicable"
                }
            )
        
        # 3. API ERROR → None + warning (not agent's fault)
        try:
            score = self._call_llm_judge(...)
        except boto3.exceptions.ClientError as e:
            return MetricScore(
                score=None,
                applicable=False,
                reasoning="Cannot evaluate: Bedrock API error",
                metadata={
                    "warning": "edge_case",
                    "edge_case_type": "api_error",
                    "error_details": str(e)
                }
            )
        
        # 4. AGENT PERFORMED POORLY → Low Score (0.0)
        return MetricScore(score=score, applicable=True, reasoning="...")
```

**Decision Matrix:**

| Situation | Return | Metadata | Rationale |
|-----------|--------|----------|-----------|
| **Missing data** (no expected_output) | **None** | `edge_case_type: "missing_data"` | Cannot evaluate |
| **Not applicable** (no tool calls for MCP) | **None** | `edge_case_type: "not_applicable"` | Nothing to evaluate |
| **API error** (AWS down) | **None** | `edge_case_type: "api_error"` | Not agent's fault |
| **Invalid data** (negative values) | **0.0** | N/A | Validation error |
| **Agent failure** (wrong answer) | **0.0** | N/A | Legitimate poor performance |
| **Empty but valid** (0 expected, 0 called) | **1.0** | N/A | Correct behavior |

**Edge Case Types:**
- `missing_data` - Required data not provided (question, response, expected_output, etc.)
- `not_applicable` - Metric can't be evaluated for this trace type (no tools, < 2 turns, not multi-agent)
- `api_error` - Infrastructure/service failure (Bedrock API, network issues)

**Key Point**: Eliminates redundant `ground_truth` existence checks - just check `expected_output` directly.

---

## Implementation: Three-Tier Error Handling

```python
class AnswerRelevanceMetric(BaseMetric):
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        trace = evaluation_input.trace
        
        # TIER 1: Validation - Required Data
        question = self._extract_question(trace)
        if question is None:
            raise ValueError(
                "Cannot evaluate answer relevance: no user question in trace. "
                "This indicates a malformed trace."
            )
        
        response = self._extract_response(trace)
        if response is None:
            raise ValueError(
                "Cannot evaluate answer relevance: no assistant response in trace. "
                "This indicates a malformed trace."
            )
        
        # TIER 2: Infrastructure - Can Retry
        try:
            score, reasoning = self._llm_judge_relevance(question, response)
        except boto3.exceptions.ClientError as e:
            raise RuntimeError(
                f"Bedrock API error: {e}. This is an infrastructure issue, not agent performance."
            ) from e
        
        # TIER 3: Evaluation - Return Score
        return MetricScore(
            metric_name=self.get_name(),
            score=score,
            reasoning=reasoning,
            applicable=True  # Successfully evaluated
        )


class MCPComplianceMetric(BaseMetric):
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        tool_calls = evaluation_input.trace.tool_calls
        
        # NOT APPLICABLE: Return None (not an error)
        if len(tool_calls) == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                applicable=False,
                reasoning="MCP compliance not applicable: no tool calls in trace"
            )
        
        # EVALUATE: Normal calculation
        score = self._calculate_mcp_compliance(tool_calls)
        return MetricScore(
            metric_name=self.get_name(),
            score=score,
            applicable=True,
            reasoning=f"MCP compliance: {score:.1%}"
        )
```

---

## Quick Wins (Can Do Today)

1. **Document limitation** in metric docstrings
2. **Add warning** for order-sensitive operations  
3. **Audit 64 edge case handlers** found in metrics
4. **Create issue** for edge case handling improvement

**Effort**: 1 hour

---

## Summary Table: Edge Cases Across All Metrics

### Tool Calling Metrics

| Metric | Edge Case | Current Score | Proposed | Issue Type |
|--------|-----------|---------------|----------|------------|
| `tool_selection_accuracy` | No tools expected, none called | 1.0 ✓ | 1.0 ✓ | OK (valid scenario) |
| | No tools expected, but called | 0.0 ✓ | 0.0 ✓ | OK |
| | Tools expected, none called | 0.0 ✓ | 0.0 ✓ | OK |
| | **Arguments swapped** | **1.0 ❌** | **0.0 ✓** | **FALSE POSITIVE** |
| | Extra parameters added | 1.0 ⚠️ | 0.25-1.0* | Overly lenient |
| `tool_sequence_correctness` | Both empty | 1.0 ✓ | 1.0 ✓ | OK |
| | One empty | 0.0 ✓ | 0.0 ✓ | OK |
| `parameter_quality` | No tools | 1.0 ⚠️ | Exception/None | Inflation |
| | **Arguments swapped** | **1.0 ❌** | **0.0 ✓** | **FALSE POSITIVE** |
| `mcp_compliance` | No tool calls | **1.0 ⚠️** | **None ✓** | **Inflation** |

*Depends on `allow_extra_params` setting

### Response Quality Metrics

| Metric | Edge Case | Current Score | Proposed | Issue Type |
|--------|-----------|---------------|----------|------------|
| `answer_relevance` | No user question | **0.0 ❌** | **Exception ✓** | **Deflation** |
| | No agent response | **0.0 ❌** | **Exception ✓** | **Deflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `completeness` | No ground truth | Exception ✓ | Exception ✓ | OK |
| | No expected output | 1.0 ⚠️ | Exception/None | Inflation |
| | No response | **0.0 ❌** | **Exception ✓** | **Deflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `hallucination_score` | No response | **1.0 ⚠️** | **Exception ✓** | **Inflation** |
| | No context | **1.0 ⚠️** | **Exception ✓** | **Inflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `accuracy` | No question | **0.0 ❌** | **Exception ✓** | **Deflation** |
| | No response | **0.0 ❌** | **Exception ✓** | **Deflation** |
| | No reference | **0.0 ❌** | **Exception ✓** | **Deflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |

### Responsible AI Metrics

| Metric | Edge Case | Current Score | Proposed | Issue Type |
|--------|-----------|---------------|----------|------------|
| `safety_score` | No response | **1.0 ⚠️** | **Exception ✓** | **Inflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `bias_score` | No response | **1.0 ⚠️** | **Exception ✓** | **Inflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `toxicity_score` | No response | **1.0 ⚠️** | **Exception ✓** | **Inflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `prompt_injection_detection` | No user messages | **1.0 ⚠️** | **None ✓** | **Inflation** |

### Performance Metrics

| Metric | Edge Case | Current Score | Proposed | Issue Type |
|--------|-----------|---------------|----------|------------|
| `latency_score` | No latency metadata | **1.0 ⚠️** | **None ✓** | **Inflation** |
| | Negative latency | 0.0 ✓ | 0.0 ✓ | OK (validation) |
| `token_efficiency` | No token metadata | **1.0 ⚠️** | **None ✓** | **Inflation** |
| `cost_efficiency` | No cost metadata | **1.0 ⚠️** | **None ✓** | **Inflation** |
| | Negative cost | 0.0 ✓ | 0.0 ✓ | OK (validation) |
| `throughput` | No throughput metadata | **1.0 ⚠️** | **None ✓** | **Inflation** |

### Multi-Turn Metrics

| Metric | Edge Case | Current Score | Proposed | Issue Type |
|--------|-----------|---------------|----------|------------|
| `context_retention` | < 4 messages (< 2 turns) | **1.0 ⚠️** | **None ✓** | **Inflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `coherence` | < 4 messages (< 2 turns) | **1.0 ⚠️** | **None ✓** | **Inflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `conversation_completeness` | No ground truth | Exception ✓ | Exception ✓ | OK |
| | No success criteria | **1.0 ⚠️** | **None ✓** | **Inflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `turn_efficiency` | 0 turns | **1.0 ⚠️** | **None ✓** | **Inflation** |

### Multi-Agent Metrics

| Metric | Edge Case | Current Score | Proposed | Issue Type |
|--------|-----------|---------------|----------|------------|
| `agent_utilization` | Not multi-agent trace | **1.0 ⚠️** | **None ✓** | **Inflation** |
| | No agent traces | **1.0 ⚠️** | **None ✓** | **Inflation** |
| | No work performed | **1.0 ⚠️** | **None ✓** | **Inflation** |
| `delegation_quality` | Not multi-agent trace | **1.0 ⚠️** | **None ✓** | **Inflation** |
| | No coordination events | **1.0 ⚠️** | **None ✓** | **Inflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `workflow_completion` | Not multi-agent trace | **1.0 ⚠️** | **None ✓** | **Inflation** |
| | Workflow FAILED status | 0.0 ✓ | 0.0 ✓ | OK (actual failure) |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `coordination_efficiency` | Not multi-agent trace | **1.0 ⚠️** | **None ✓** | **Inflation** |
| | No work performed | **1.0 ⚠️** | **None ✓** | **Inflation** |

### Reasoning Metrics

| Metric | Edge Case | Current Score | Proposed | Issue Type |
|--------|-----------|---------------|----------|------------|
| `chain_of_thought_coherence` | No reasoning found | **1.0 ⚠️** | **Exception ✓** | **Inflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `logical_consistency` | No reasoning found | **1.0 ⚠️** | **Exception ✓** | **Inflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `reasoning_step_correctness` | No reasoning found | **1.0 ⚠️** | **Exception ✓** | **Inflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |
| `fallacy_detection` | No reasoning found | **1.0 ⚠️** | **Exception ✓** | **Inflation** |
| | Bedrock API error | **0.0 ❌** | **Exception ✓** | **Deflation** |

---

## Statistics Summary

**Total Metrics**: 28  
**Edge Case Handlers**: 64 instances

**Issues Found:**

| Issue Type | Count | Examples |
|------------|-------|----------|
| **Inflation** (returns 1.0) | 23 | MCP compliance, no context, no turns, no data |
| **Deflation** (returns 0.0) | 16 | API errors, missing questions, missing responses |
| **False Positives** | 2 | Swapped arguments in tool/parameter metrics |
| **Valid Behavior** | 23 | Empty expected=empty called, validation errors |

**Critical**: 37 edge cases (58%) need fixes (21 inflation + 16 deflation)!

**Note**: Numbers updated after identifying valid cases (both empty = 1.0 is correct behavior)

---

## Impact on Sample Evaluation

### Scenario: Single-turn Q&A Agent (No Tools)

```python
Trace: 
  - User: "What is Paris?"
  - Agent: "Paris is the capital of France."
  - Tool calls: []
  - Turns: 1
```

#### Current Scoring (Inflated)

| Metric | Score | Issue |
|--------|-------|-------|
| tool_selection_accuracy | 1.0 | ⚠️ Inflation (no tools) |
| tool_sequence_correctness | 1.0 | ⚠️ Inflation (no tools) |
| parameter_quality | 1.0 | ⚠️ Inflation (no tools) |
| mcp_compliance | **1.0** | ⚠️ **Inflation** |
| answer_relevance | 0.85 | ✓ Valid |
| context_retention | **1.0** | ⚠️ **Inflation (< 2 turns)** |
| coherence | **1.0** | ⚠️ **Inflation (< 2 turns)** |
| turn_efficiency | **1.0** | ⚠️ **Inflation (1 turn)** |

**Average: 0.98** (artificially high!)

#### Proposed Scoring (Accurate)

| Metric | Score | Change |
|--------|-------|--------|
| tool_selection_accuracy | None | Filtered (not applicable) |
| tool_sequence_correctness | None | Filtered (not applicable) |
| parameter_quality | None | Filtered (not applicable) |
| mcp_compliance | None | Filtered (not applicable) |
| answer_relevance | 0.85 | ✓ Evaluated |
| context_retention | None | Filtered (< 2 turns) |
| coherence | None | Filtered (< 2 turns) |
| turn_efficiency | None | Filtered (only 1 turn) |

**Average: 0.85** (accurate - only applicable metric!)

**Impact**: Score dropped from 0.98 to 0.85 due to removing inflated metrics.

---

## Token/Cost Efficiency Metrics - Circular Dependency Issue

### Problem: Quality Score Not Calculated

Both `TokenEfficiencyMetric` and `CostEfficiencyMetric` need a **quality score** to calculate efficiency, but it's **never set**:

```python
# src/uaef/metrics/performance.py, line 154
quality_score = trace.metadata.get("quality_score", 0.5)  # ❌ Always defaults to 0.5!

# Calculate efficiency: quality per 1000 tokens
efficiency = quality_score / max(1.0, tokens_per_k)
```

**Comment in code:**
```python
# Get quality score from metadata (if available from previous evaluation)
# This would typically be set by the evaluation engine
```

### The Problem: It's NEVER Set!

Searched entire codebase: **`quality_score` is never written to `trace.metadata`** anywhere.

**Result**: 
- All agents get the same quality assumption (0.5)
- Token/cost efficiency scores are meaningless
- Cannot distinguish efficient vs inefficient agents

### Example Impact

```python
Agent A: quality=0.9 (excellent), tokens=1000
Agent B: quality=0.3 (poor), tokens=1000

Current Scores (both use 0.5):
  Agent A: 0.5 / 1.0 = 0.5
  Agent B: 0.5 / 1.0 = 0.5  # Same score despite different quality!

Correct Scores:
  Agent A: 0.9 / 1.0 = 0.9
  Agent B: 0.3 / 1.0 = 0.3  # Correctly shows difference
```

### Proposed Solution: Calculate Quality Score First

```python
class SingleAgentEvaluator:
    def evaluate(self, trace, ground_truth, ...) -> EvaluationResult:
        # 1. Calculate response quality metrics first
        quality_metrics = [
            "answer_relevance",
            "completeness", 
            "accuracy",
            "hallucination_score"
        ]
        
        quality_scores = self._calculate_metrics(quality_metrics, eval_input)
        avg_quality = sum(s.score for s in quality_scores if s.score) / len(quality_scores)
        
        # 2. Set quality score in trace metadata
        trace.metadata["quality_score"] = avg_quality
        
        # 3. Now calculate efficiency metrics (they can access quality_score)
        efficiency_metrics = ["token_efficiency", "cost_efficiency"]
        efficiency_scores = self._calculate_metrics(efficiency_metrics, eval_input)
        
        # 4. Aggregate all results
        all_scores = quality_scores + efficiency_scores
        ...
```

**Alternative: Pass Quality Score as Parameter**

```python
class TokenEfficiencyMetric(BaseMetric):
    def calculate(
        self, 
        evaluation_input: EvaluationInput,
        quality_score: Optional[float] = None  # NEW parameter
    ) -> MetricScore:
        # Try to get from parameter first, then metadata, then default
        quality = quality_score or trace.metadata.get("quality_score", 0.5)
        
        # Calculate efficiency
        efficiency = quality / max(1.0, tokens_per_k)
        ...
```

### Current Formula Issues

**Current implementation:**
```python
tokens_per_k = total_tokens / 1000.0
efficiency = quality_score / max(1.0, tokens_per_k)
score = max(0.0, min(1.0, efficiency))
```

**Problems:**

| Issue | Current Behavior | Impact |
|-------|------------------|--------|
| **Flat below threshold** | 100 tokens = 999 tokens = same score | No reward for efficiency |
| **Harsh linear decay** | 2000 tokens = 50% score drop | Penalizes complex tasks |
| **Fixed baseline (1000)** | All tasks same threshold | Not task-appropriate |

### Proposed Formula: Threshold + Configurable Penalty Modes

```python
class TokenEfficiencyMetric(BaseMetric):
    def __init__(
        self, 
        expected_tokens: int = 1000,
        penalty_mode: str = "light"  # "light" or "heavy"
    ):
        """
        expected_tokens: Threshold for token usage
        penalty_mode: "light" (sqrt) or "heavy" (linear) for exceeding threshold
        """
        self.expected_tokens = expected_tokens
        self.penalty_mode = penalty_mode
    
    def calculate_efficiency(self, tokens, quality):
        if tokens <= self.expected_tokens:
            # At or below threshold: full quality score (reward efficiency)
            return quality
        else:
            # Above threshold: apply penalty based on mode
            ratio = tokens / self.expected_tokens
            
            if self.penalty_mode == "light":
                # Square root penalty (gentler)
                efficiency = quality / (ratio ** 0.5)
            else:  # "heavy"
                # Linear penalty (stricter)
                efficiency = quality / ratio
            
            return max(0.0, min(1.0, efficiency))
```

**Results Comparison** (quality=0.8, expected=1000):

| Tokens | Current (linear) | Light (√) | Heavy (linear) | Notes |
|--------|-----------------|-----------|----------------|-------|
| 100 | 0.80 | **0.80** | **0.80** | ✅ All flat below threshold |
| 500 | 0.80 | **0.80** | **0.80** | ✅ Rewards efficiency |
| 1000 | 0.80 | **0.80** | **0.80** | ✅ All match at threshold |
| 1500 | 0.53 | **0.65** ⬆️ | 0.53 | Light 23% less harsh |
| 2000 | 0.40 | **0.57** ⬆️ | 0.40 | Light 42% less harsh |
| 5000 | 0.16 | **0.36** ⬆️ | 0.16 | Light 125% less harsh |
| 10000 | 0.08 | **0.25** ⬆️ | 0.08 | Light 213% less harsh |

**Mode Recommendations:**

- **Light mode (√ penalty)**: For complex tasks, research agents, document analysis
  - Gentler on legitimate token usage
  - 2x tokens = √2 ≈ 1.41x penalty (vs 2x in linear)
  
- **Heavy mode (linear penalty)**: For simple tasks, API wrappers, quick lookups
  - Strict efficiency requirements
  - 2x tokens = 2x penalty

**Benefits:**
- ✅ Flat below threshold (no penalty for efficiency)
- ✅ User chooses penalty strictness
- ✅ Scores stay in [0, 1]
- ✅ Task-appropriate thresholds

### Cost Efficiency - Same Issues + Critical Bug

`CostEfficiencyMetric` has the **same issues** as token efficiency, **plus a bug**:

```python
# src/uaef/metrics/performance.py, line 240
cost_normalized = cost_usd / 0.01
efficiency = quality_score / max(0.01, cost_normalized)  # ❌ BUG: should be 1.0!
score = max(0.0, min(1.0, efficiency))
```

**The Bug: Wrong Threshold**

| Formula | Normalization | Threshold | Flat Region | Correct? |
|---------|---------------|-----------|-------------|----------|
| **Token** | `tokens / 1000` | `max(1.0, ...)` | ≤ 1000 tokens | ✓ |
| **Cost (current)** | `cost / 0.01` | `max(0.01, ...)` | ≤ **$0.0001** | ❌ **Wrong!** |
| **Cost (fixed)** | `cost / 0.01` | `max(1.0, ...)` | ≤ **$0.01** | ✓ |

**Impact of Bug:**
```
Cost $0.0001 → efficiency = 0.8 / 0.01 = 80.0 → clamped to 1.0 ✓ (masked by clamp!)
Cost $0.001  → efficiency = 0.8 / 0.10 = 8.0  → clamped to 1.0 ✓ (masked!)
Cost $0.005  → efficiency = 0.8 / 0.50 = 1.6  → clamped to 1.0 ✓ (masked!)
Cost $0.01   → efficiency = 0.8 / 1.00 = 0.8  → 0.8 ✓
```

The **clamping masked the bug** - without `min(1.0, ...)`, scores would be 8.0, 80.0, etc.!

### Proposed Fix: Same as Token Efficiency

```python
class CostEfficiencyMetric(BaseMetric):
    def __init__(
        self, 
        expected_cost: float = 0.01,
        penalty_mode: str = "light"  # "light" or "heavy"
    ):
        self.expected_cost = expected_cost
        self.penalty_mode = penalty_mode
    
    def calculate_efficiency(self, cost, quality):
        if cost <= self.expected_cost:
            # At or below expected: full quality score
            return quality
        else:
            # Above expected: apply penalty
            ratio = cost / self.expected_cost
            
            if self.penalty_mode == "light":
                efficiency = quality / (ratio ** 0.5)  # Square root
            else:  # "heavy"
                efficiency = quality / ratio  # Linear
            
            return max(0.0, min(1.0, efficiency))
```

**Clamping Note**: With corrected threshold (`max(1.0, ...)`), clamping becomes mathematically redundant but keep for defensive programming.

---

### Recommendation: Add Penalty Modes to Both

```python
# Token Efficiency
TokenEfficiencyMetric(expected_tokens=1000, penalty_mode="light")

# Cost Efficiency (same API)
CostEfficiencyMetric(expected_cost=0.01, penalty_mode="light")
```

**Usage:**
```python
# For research agents (complex tasks, allow higher resource usage)
token_metric = TokenEfficiencyMetric(expected_tokens=3000, penalty_mode="light")
cost_metric = CostEfficiencyMetric(expected_cost=0.05, penalty_mode="light")

# For API wrappers (simple tasks, strict efficiency)
token_metric = TokenEfficiencyMetric(expected_tokens=500, penalty_mode="heavy")
cost_metric = CostEfficiencyMetric(expected_cost=0.005, penalty_mode="heavy")
```

**Files to update:**
- `src/uaef/evaluation/single_agent.py` (quality score calculation)
- `src/uaef/metrics/performance.py` (TokenEfficiencyMetric + CostEfficiencyMetric)

---

## Context Retention - False Positive for Non-Retention Conversations

### Problem: Scores High Even When No Context Needs Retention

**Current UAEF Implementation:**
```python
# If < 4 messages: return 1.0
# Otherwise: send entire conversation to LLM judge
```

**Issue**: Conversation with **independent questions** still gets high scores:

```
User: "What's 2+2?"
Assistant: "4"
User: "What's 3+3?"  ← No context retention needed
Assistant: "6"

UAEF: Sends to LLM → LLM says "good conversation flow" → Score: 0.8-1.0 ❌
Should: Return None (context retention not applicable) ✓
```

### DeepEval's Superior Approach: Knowledge Extraction

**DeepEval's `KnowledgeRetentionMetric`:**

1. **Extract knowledge** from each user message:
   ```
   User: "I live in Paris" → Knowledge: {"Location": "Paris"}
   User: "What's 2+2?"     → Knowledge: {} (no facts about user)
   ```

2. **Check retention** in assistant responses:
   ```
   Assistant: "Since you live in Paris..." ← Uses prior knowledge ✓
   Assistant: "Where do you live?"        ← Asks for known info ✗
   ```

3. **Score based on verdicts**:
   ```python
   retention_count = verdicts where verdict == "no" (retained correctly)
   score = retention_count / number_of_verdicts
   
   if number_of_verdicts == 0:
       return 0  # Nothing to retain → score 0 (not 1.0!)
   ```

**Key Insight**: DeepEval returns **0** (not 1.0) when there's no knowledge to retain.

### Comparison

| Scenario | UAEF | DeepEval | Winner |
|----------|------|----------|--------|
| **Independent questions** | 0.8-1.0 ❌ (false positive) | 0 ✓ (nothing to retain) | DeepEval |
| **User provides facts, agent remembers** | 0.9 ✓ | 1.0 ✓ | Both |
| **User provides facts, agent forgets** | 0.3 ✓ | 0.0 ✓ | Both |
| **< 2 turns** | 1.0 ❌ (inflation) | None/Exception ✓ | DeepEval |

### Proposed Improvements

**Option 1: Adopt DeepEval's Extraction Approach** (Complex, 2-3 weeks)
- Extract knowledge from user messages
- Check each assistant response against accumulated knowledge
- Most accurate

**Option 2: Update LLM Prompt to Return None** (Simple, 1 day) ⭐ **Recommended**

```python
user_prompt = """
You are an AI evaluator assessing context retention in multi-turn conversations.

IMPORTANT: First determine if this conversation requires context retention.
Context retention is required when:
- User provides personal information (name, location, preferences)
- User makes references that require remembering earlier context
- Earlier turns establish facts that should be used in later turns

If NO context retention is required (e.g., independent factual questions), return:
{{"score": null, "reasoning": "Not applicable - no context retention required"}}

If context retention IS required, evaluate and return score 0-1.

[Conversation History]
{conversation}
"""
```

Then in metric code:
```python
response_json = json.loads(response_body.get('content')[0]['text'])
response_score = response_json.get("score")  # Can be None now

if response_score is None:
    return MetricScore(
        metric_name=self.get_name(),
        score=None,
        applicable=False,
        reasoning=response_json.get("reasoning", "Not applicable")
    )

return MetricScore(score=response_score, applicable=True, ...)
```

**Option 3: Adopt DeepEval's Extraction** (Complex, 2-3 weeks)
- Most accurate but requires significant redesign
- Extract knowledge, check retention per turn

---

## Conversation Completeness - Make Success Criteria Optional

### Current Limitation

**UAEF requires explicit success criteria:**
```python
if not evaluation_input.ground_truth:
    raise ValueError("Ground truth is required")

if not success_criteria:
    return MetricScore(score=1.0, ...)  # ❌ Inflation
```

### DeepEval's Approach: Automatic Intention Extraction

DeepEval **extracts intentions from conversation** when no explicit criteria provided:
1. Uses LLM to identify user intentions
2. Checks if each intention was satisfied
3. Score = satisfied / total_intentions

**DeepEval Issue**: Extracts ALL intentions including superseded ones.

### Proposed Solution: Make Success Criteria Optional ⭐

```python
class ConversationCompletenessMetric(BaseMetric):
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        # Get success criteria from ground truth (if provided)
        success_criteria = None
        if evaluation_input.ground_truth:
            success_criteria = evaluation_input.ground_truth.expected_output
        
        if success_criteria:
            # MODE 1: Use explicit criteria (when provided)
            score = self._evaluate_with_criteria(conversation, success_criteria)
        else:
            # MODE 2: Extract final intentions (when not provided)
            intentions = self._extract_final_intentions(conversation)
            
            if not intentions:
                return MetricScore(
                    score=None,
                    applicable=False,
                    reasoning="No user intentions detected in conversation"
                )
            
            score = self._evaluate_intentions(conversation, intentions)
        
        return MetricScore(score=score, applicable=True, ...)
```

**Intention Extraction Prompt** (improved from DeepEval):
```python
prompt = """
Extract the user's FINAL intentions from this conversation.

CRITICAL: If the user changes their mind or modifies their goal:
  - ONLY extract the final/current intention
  - DO NOT include superseded or canceled intentions

Examples:
  - "Book Paris" → later "Change to London" → Extract: ["Book flight to London"]
  - "I want pizza" → later "Cancel that" → Extract: [] (canceled)

Conversation:
{conversation}

JSON:
{{"intentions": ["final intention 1", ...]}}
"""
```

**Benefits:**
- ✅ Works with or without explicit criteria
- ✅ Handles evolving goals correctly
- ✅ No inflation when criteria missing
- ✅ Backward compatible (explicit criteria still works)

**Usage:**
```python
# With explicit criteria (current)
ground_truth = GroundTruth(expected_output="Book flight to London")
result = evaluate(trace, ground_truth)

# Without criteria (new - automatic extraction)
result = evaluate(trace, ground_truth=None)  # Extracts intentions automatically
```

---

## Turn Efficiency - Incorporate Completeness

### Problem: Current Formula Ignores Goal Achievement

**Current Implementation:**
```python
def __init__(self, expected_turns: int = 5):
    self.expected_turns = expected_turns

if turn_count <= self.expected_turns:
    score = 1.0  # ❌ Always 1.0, ignores if goal achieved!
else:
    score = self.expected_turns / turn_count  # ❌ Ignores completeness!
```

**Issue**: Treats all conversations within threshold as equally efficient, regardless of whether goal was achieved.

**Example:**
```
Conversation A: 3 turns, achieved goal (completeness=1.0) → score: 1.0
Conversation B: 3 turns, didn't achieve goal (completeness=0.3) → score: 1.0 ❌ SAME!
```

### Proposed Solution: Keep Threshold + Add Completeness ⭐

```python
class TurnEfficiencyMetric(BaseMetric):
    def __init__(self, expected_turns: int = 5):
        self.expected_turns = expected_turns
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        turn_count = len(trace.messages) // 2
        
        if turn_count == 0:
            return MetricScore(score=None, applicable=False,
                             reasoning="No turns to evaluate")
        
        # Get conversation completeness score
        completeness_score = self._get_completeness_score(evaluation_input)
        
        if completeness_score is None:
            return MetricScore(score=None, applicable=False,
                             reasoning="Cannot evaluate efficiency without completeness")
        
        # Incorporate completeness into efficiency
        if turn_count <= self.expected_turns:
            # Within threshold: efficiency = completeness (achieved goal efficiently)
            efficiency = completeness_score
        else:
            # Beyond threshold: efficiency = completeness * (expected / actual)
            efficiency = completeness_score * (self.expected_turns / turn_count)
        
        return MetricScore(
            metric_name=self.get_name(),
            score=efficiency,
            reasoning=f"Completeness {completeness_score:.2f} in {turn_count} turns (expected ≤{self.expected_turns})",
            metadata={
                "turn_count": turn_count,
                "expected_turns": self.expected_turns,
                "completeness": completeness_score
            }
        )
```

**Formula:**
```
if turn_count ≤ expected:
    efficiency = completeness
else:
    efficiency = completeness × (expected / actual)
```

**Examples:**

| Completeness | Turns | Expected | Current | Proposed | Difference |
|--------------|-------|----------|---------|----------|------------|
| **1.0** | 3 | 5 | 1.0 | **1.0** | Same (efficient) |
| **0.3** | 3 | 5 | 1.0 | **0.3** | ✅ Detects incomplete! |
| **1.0** | 10 | 5 | 0.5 | **0.5** | Same |
| **0.5** | 10 | 5 | 0.5 | **0.25** | ✅ Lower (incomplete + many turns) |

**Benefits:**
- ✅ Keeps threshold (task-appropriate expected turns)
- ✅ Factors in goal achievement (completeness)
- ✅ Penalizes incomplete conversations
- ✅ Rewards completing goals in fewer turns
- ✅ Distinguishes efficient-and-complete from efficient-but-incomplete

**Key Improvement**: Within threshold, efficiency = completeness (not always 1.0!)

**Implementation Note:** Requires calculating `ConversationCompletenessMetric` first.

---

## Conversation Completeness - Make Success Criteria Optional

### Proposed: Optional Criteria with Intention Extraction

```python
if success_criteria:
    # Use explicit criteria (when provided)
    score = self._evaluate_with_criteria(conversation, success_criteria)
else:
    # Extract final intentions (when not provided)
    intentions = self._extract_final_intentions(conversation)
    if not intentions:
        return MetricScore(score=None, applicable=False,
                         reasoning="No user intentions detected")
    score = self._evaluate_intentions(conversation, intentions)
```

**Intention Extraction** (improved from DeepEval):
```
CRITICAL: If user changes their mind, ONLY extract final intention.
  - "Book Paris" → "Change to London" → Extract: ["Book London"]
  - "I want pizza" → "Cancel that" → Extract: []
```

**Benefits:**
- ✅ Works with or without explicit criteria
- ✅ Doesn't extract superseded intentions
- ✅ Backward compatible

---

## Agent Utilization - Single vs Batch Evaluation

### Current: Single-Query Utilization

**Current Implementation:**
```python
# Calculates coefficient of variation (CV) for work distribution in single query
mean_work = total_work / num_agents
cv = std_dev / mean_work
score = exp(-cv)  # Lower CV = more even = higher score
```

**Works but has limitations** - single query utilization depends on task requirements.

### Recommendation: Add Batch Utilization ⭐

**Keep single-query calculation, ADD batch calculation:**

```python
class AgentUtilizationMetric(BaseMetric):
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        # Single query utilization (current)
        multi_trace = evaluation_input.multi_agent_trace
        single_util = self._calculate_single_utilization(multi_trace)
        
        return MetricScore(
            metric_name=self.get_name(),
            score=single_util,
            reasoning=f"Single-query utilization: {single_util:.2f}",
            metadata={
                "utilization_type": "single",
                "note": "Batch utilization is more meaningful for detecting systematic underutilization"
            }
        )
    
    def calculate_batch(
        self, 
        multi_traces: List[MultiAgentTrace]
    ) -> MetricScore:
        """
        Calculate utilization across batch of queries.
        More meaningful - detects agents consistently underutilized.
        """
        # Aggregate work across all queries
        total_work_per_agent = {}
        for trace in multi_traces:
            for agent_id, work in self._get_work_per_agent(trace).items():
                total_work_per_agent[agent_id] = total_work_per_agent.get(agent_id, 0) + work
        
        # Calculate CV across batch
        mean_work = sum(total_work_per_agent.values()) / len(total_work_per_agent)
        variance = sum((w - mean_work)**2 for w in total_work_per_agent.values()) / len(total_work_per_agent)
        cv = math.sqrt(variance) / mean_work
        score = math.exp(-cv)
        
        return MetricScore(
            metric_name=f"{self.get_name()}_batch",
            score=score,
            reasoning=f"Batch utilization across {len(multi_traces)} queries",
            metadata={
                "utilization_type": "batch",
                "query_count": len(multi_traces),
                "agent_work_totals": total_work_per_agent
            }
        )
```

**Why Batch is More Meaningful:**

| Evaluation Type | What It Shows | Use Case |
|-----------------|---------------|----------|
| **Single query** | Distribution for this specific task | Debug single execution |
| **Batch queries** | Systematic under/over-utilization | Identify unused agents, bottlenecks |

**Example:**
```
Single query: research=20, summary=5 → CV high, but task-appropriate
Batch (100 queries): research=2000, summary=50 → Summary agent underutilized! ✓
```

**Documentation Note:**
```
Agent utilization can be calculated for single queries, but is most meaningful
when evaluated across a batch of queries to identify systematic patterns.

Single query: Shows distribution for that specific task
Batch queries: Reveals consistently underutilized or overloaded agents
```

---

## Workflow Completion - Binary Failure Handling

### UAEF vs DeepEval: Binary vs Granular

| Aspect | UAEF Workflow Completion | DeepEval Task Completion |
|--------|-------------------------|-------------------------|
| **Scope** | Multi-agent only | Single or multi-agent |
| **Task source** | Explicit (required) | Explicit OR extracted from trace |
| **Workflow failure** | **Binary: FAILED → 0.0** | **Granular: partial credit** |
| **Trace usage** | Summary (status, events, outputs) | Deep (spans, tools, all actions) |
| **Edge: no criteria** | Returns 1.0 ❌ | Extracts task ✓ |

### Critical Difference: Handling Workflow Failure

**Scenario:** Workflow fails after 70% completion

**UAEF:**
```python
if workflow_status == WorkflowStatus.FAILED:
    return MetricScore(score=0.0, reasoning="Workflow failed")
```
- Score: **0.0** (no credit for partial work)
- Binary: success or total failure

**DeepEval:**
```python
# Extracts what WAS achieved
outcome = "Completed steps 1-7 of 10, failed at step 8"
# Evaluates alignment with task
score = llm_judge_alignment(task, outcome)
```
- Score: **0.7** (credit for 70% completion)
- Granular: reflects partial progress

### Proposed Improvement: Evaluate Partial Completion

```python
class WorkflowCompletionMetric(BaseMetric):
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        if not multi_agent_trace:
            return MetricScore(score=None, applicable=False,
                             reasoning="Not a multi-agent workflow")
        
        # Get or extract success criteria
        success_criteria = (
            ground_truth.expected_output if ground_truth
            else self._extract_task_from_workflow(multi_trace)
        )
        
        if not success_criteria:
            return MetricScore(score=None, applicable=False,
                             reasoning="No task/criteria to evaluate")
        
        # Extract actual outcome (even if FAILED)
        actual_outcome = self._extract_workflow_outcome(multi_trace)
        
        # Evaluate alignment (partial credit possible)
        alignment_score = self._llm_judge_alignment(success_criteria, actual_outcome)
        
        # Apply workflow status penalty (not binary)
        if workflow_status == WorkflowStatus.FAILED:
            # 50% penalty for failure, but keep partial completion credit
            final_score = alignment_score * 0.5
        else:
            final_score = alignment_score
        
        return MetricScore(
            metric_name=self.get_name(),
            score=final_score,
            reasoning=f"Task alignment: {alignment_score:.2f}, Status: {workflow_status}",
            metadata={
                "alignment_score": alignment_score,
                "workflow_status": workflow_status.value,
                "partial_completion": workflow_status == WorkflowStatus.FAILED
            }
        )
```

**Benefits:**
- ✅ Credit partial work (70% done = 0.35 score, not 0.0)
- ✅ Optional success criteria (extract if not provided)
- ✅ Return None for non-applicable (not 1.0)
- ✅ Non-binary evaluation

**Example Scores:**

| Completion | Status | Alignment | Penalty | Final Score |
|------------|--------|-----------|---------|-------------|
| 100% | COMPLETED | 1.0 | None | **1.0** |
| 70% | FAILED | 0.7 | 0.5x | **0.35** (was 0.0) |
| 30% | FAILED | 0.3 | 0.5x | **0.15** (was 0.0) |
| 90% | COMPLETED | 0.9 | None | **0.9** |

---

## Coordination Efficiency - Hardcoded Ideal Ratio

### Problem: Fixed Ratio for All Workflow Types

**Current Implementation:**
```python
# Line 623 - hardcoded!
ideal_ratio = 0.1  # 1 coordination per 10 work units

if coordination_ratio <= ideal_ratio:
    score = 1.0
else:
    score = exp(-(coordination_ratio - ideal_ratio) * 5)
```

**Issue**: Different workflow patterns need different coordination levels:

| Workflow Type | Coordination Needs | Ideal Ratio | Current Score |
|---------------|-------------------|-------------|---------------|
| **Hierarchical** (1 orchestrator) | Low | ~0.05 | Penalized if > 0.1 |
| **Collaborative** (peer-to-peer) | Medium | ~0.15 | Penalized if > 0.1 ❌ |
| **Sequential pipeline** | High | ~0.25 | Heavily penalized ❌ |
| **Event-driven** | Variable | Depends | May be wrong |

**Example:**
```
Sequential workflow: research → analyze → summarize → format
  - Each step requires handoff coordination
  - 40 work units, 8 coordination events
  - Ratio: 8/40 = 0.2
  - Current score: exp(-(0.2-0.1)*5) = exp(-0.5) = 0.61
  
Is 0.61 bad? Not necessarily - sequential workflows naturally need more coordination!
```

### Proposed Solution: Make Ideal Ratio Configurable

```python
class CoordinationEfficiencyMetric(BaseMetric):
    def __init__(self, ideal_ratio: float = 0.1, decay_rate: float = 5.0):
        """
        Args:
            ideal_ratio: Target coordination-to-work ratio
                - 0.05: Hierarchical (centralized orchestrator)
                - 0.10: Balanced (default)
                - 0.20: Collaborative (peer-to-peer)
                - 0.25: Sequential pipeline (many handoffs)
            decay_rate: How quickly score decays above ideal (default: 5.0)
        """
        self.ideal_ratio = ideal_ratio
        self.decay_rate = decay_rate
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        # ... calculate coordination_ratio ...
        
        if coordination_ratio <= self.ideal_ratio:
            score = 1.0
            reasoning = f"Efficient coordination: ratio {coordination_ratio:.2f} ≤ ideal {self.ideal_ratio}"
        else:
            import math
            score = math.exp(-(coordination_ratio - self.ideal_ratio) * self.decay_rate)
            score = max(0.0, min(1.0, score))
            reasoning = f"Coordination overhead: ratio {coordination_ratio:.2f} > ideal {self.ideal_ratio}"
        
        return MetricScore(
            score=score,
            reasoning=reasoning,
            metadata={
                "coordination_ratio": coordination_ratio,
                "ideal_ratio": self.ideal_ratio,
                "decay_rate": self.decay_rate
            }
        )
```

**Usage by Workflow Pattern:**
```python
# Hierarchical workflow (centralized orchestrator)
metric = CoordinationEfficiencyMetric(ideal_ratio=0.05)

# Collaborative workflow (peer-to-peer)
metric = CoordinationEfficiencyMetric(ideal_ratio=0.20)

# Sequential pipeline (many handoffs)
metric = CoordinationEfficiencyMetric(ideal_ratio=0.25, decay_rate=3.0)  # Gentler
```

**Benefits:**
- ✅ Workflow-type appropriate thresholds
- ✅ Configurable decay rate (strict vs lenient)
- ✅ No false penalties for high-coordination workflows
- ✅ Clear in metadata what ideal was used

### Additional: Document in Ground Truth

```python
ground_truth = GroundTruth(
    expected_output="Process all orders",
    metadata={
        "workflow_pattern": "sequential_pipeline",
        "ideal_coordination_ratio": 0.25
    }
)
```

**Recommendation:** Make `ideal_ratio` configurable parameter, document typical values for different workflow patterns.

---

## Missing Metrics - Use DeepEval Integration

### Metrics Missing in UAEF

| DeepEval Metric | What It Evaluates | Similar UAEF Metric? | Key Difference | Why Missing |
|-----------------|------------------|---------------------|----------------|-------------|
| **Plan Quality** | Quality of agent's plan (completeness, logic, optimality) | Reasoning coherence | UAEF checks reasoning flow, NOT plan quality specifically | Gap for planning agents |
| **Plan Adherence** | Did execution follow stated plan? | Tool sequence correctness | UAEF checks tool order vs ground truth, NOT plan vs execution | Gap for plan compliance |
| **Step Efficiency** | Qualitative: Are steps minimal? Detects redundancy | Turn efficiency | UAEF counts turns (quantitative), DeepEval judges path quality (qualitative) | Different concept |
| **Knowledge Retention** | Remembers user facts (name, location, etc.) | Context retention | UAEF checks general context use, DeepEval extracts & tracks specific user facts | More granular |
| **Turn Relevancy** | Per-turn: Is each response relevant? | Coherence | UAEF checks overall flow, DeepEval checks each turn independently | Different granularity |

### Detailed Comparison

#### Plan Quality vs Reasoning Coherence
- **UAEF (Coherence)**: "Does reasoning flow logically?" - checks connections between reasoning steps
- **DeepEval (Plan Quality)**: "Is the plan complete and optimal?" - checks if plan addresses all requirements
- **Difference**: Coherence = flow quality, Plan Quality = plan completeness

#### Plan Adherence vs Tool Sequence Correctness
- **UAEF (Tool Sequence)**: Compares actual tool calls vs **expected (ground truth)**
- **DeepEval (Plan Adherence)**: Compares actual execution vs **agent's own plan**
- **Difference**: External expectation vs self-compliance

#### Step Efficiency vs Turn Efficiency
- **UAEF (Turn Efficiency)**: Counts turns, compares to threshold (quantitative)
- **DeepEval (Step Efficiency)**: LLM judges if steps were necessary (qualitative)
- **Difference**: "How many?" vs "Were they needed?"

#### Knowledge Retention vs Context Retention
- **UAEF (Context Retention)**: Holistic - "Does agent use earlier context appropriately?"
- **DeepEval (Knowledge Retention)**: Granular - Extracts user facts, checks if each fact is remembered
- **Difference**: General context use vs specific fact tracking

#### Turn Relevancy vs Coherence
- **UAEF (Coherence)**: Entire conversation - "Does conversation flow logically?"
- **DeepEval (Turn Relevancy)**: Per-turn - "Is this response relevant to user's message?"
- **Difference**: Holistic conversation flow vs per-turn relevance checks

### New Metric Suggestion: Error Recovery & Adaptation

**Metric Name**: `error_recovery_quality`

**Purpose**: Evaluate if agent handles failures and adapts to errors appropriately.

**What It Detects:**
- ❌ Repeatedly trying same failed approach
- ❌ Ignoring tool call errors
- ❌ Not adapting when results are unexpected
- ✅ Trying alternative approaches after failure
- ✅ Gracefully handling errors
- ✅ Learning from failed attempts

**Algorithm:**
```python
class ErrorRecoveryMetric(BaseMetric):
    """
    Evaluates agent's ability to handle failures and adapt.
    Detects repeated failed attempts and lack of error handling.
    """
    
    def get_dimension(self) -> str:
        return "Tool Calling"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        trace = evaluation_input.trace
        tool_calls = trace.tool_calls
        
        # Analyze tool calls for failure patterns
        failures = self._detect_failures(tool_calls)
        
        if not failures:
            return MetricScore(
                score=None,
                applicable=False,
                reasoning="No failures detected to evaluate recovery"
            )
        
        # Check for repeated failures (same tool, same params)
        repeated_failures = self._detect_repeated_failures(tool_calls)
        
        # Check if agent adapted after failures
        adaptations = self._detect_adaptations(tool_calls, trace.messages)
        
        # Score based on adaptation rate
        if len(repeated_failures) > 0:
            # Penalize repeated failures
            score = max(0.0, 1.0 - (len(repeated_failures) / len(failures)))
        else:
            # No repeated failures - check if adapted
            adaptation_rate = len(adaptations) / len(failures)
            score = adaptation_rate
        
        return MetricScore(
            metric_name=self.get_name(),
            score=score,
            reasoning=f"{len(repeated_failures)} repeated failures, {len(adaptations)} adaptations",
            metadata={
                "total_failures": len(failures),
                "repeated_failures": repeated_failures,
                "adaptations": adaptations
            }
        )
    
    def _detect_failures(self, tool_calls: List[ToolCall]) -> List[ToolCall]:
        """Detect failed tool calls from results."""
        failures = []
        for tc in tool_calls:
            if tc.result:
                # Check for common failure indicators
                result_str = str(tc.result).lower()
                if any(indicator in result_str for indicator in 
                       ["error", "failed", "exception", "timeout", "null", "none"]):
                    failures.append(tc)
        return failures
    
    def _detect_repeated_failures(self, tool_calls: List[ToolCall]) -> List[Dict]:
        """Detect same tool called multiple times with same params after failure."""
        repeated = []
        for i, tc in enumerate(tool_calls):
            if self._is_failure(tc):
                # Check if same call appears again after this failure
                for j in range(i+1, len(tool_calls)):
                    if (tool_calls[j].name == tc.name and 
                        tool_calls[j].arguments == tc.arguments):
                        repeated.append({
                            "tool": tc.name,
                            "params": tc.arguments,
                            "attempts": [i, j]
                        })
        return repeated
    
    def _detect_adaptations(self, tool_calls: List[ToolCall], 
                           messages: List[Message]) -> List[Dict]:
        """Detect if agent tried alternative approaches after failure."""
        adaptations = []
        for i, tc in enumerate(tool_calls):
            if self._is_failure(tc) and i+1 < len(tool_calls):
                next_tc = tool_calls[i+1]
                # Check if changed approach
                if next_tc.name != tc.name or next_tc.arguments != tc.arguments:
                    adaptations.append({
                        "failed": tc.name,
                        "adapted_to": next_tc.name,
                        "strategy": "changed_tool" if next_tc.name != tc.name else "changed_params"
                    })
        return adaptations
```

**Examples:**

| Scenario | Behavior | Score | Issue |
|----------|----------|-------|-------|
| **Good recovery** | Tool fails → tries alternative tool | 1.0 | ✅ Adapted |
| **Good recovery** | Tool fails → retries with different params | 1.0 | ✅ Adapted |
| **Poor recovery** | Tool fails → retries same tool, same params 3x | 0.0 | ❌ Stuck |
| **Poor recovery** | Tool fails → ignores error, continues | 0.3 | ❌ No recovery |
| **No failures** | All tools succeed | None | Not applicable |

**Edge Cases:**
- No failures → return None (nothing to evaluate)
- No tool calls → return None (not applicable)

**Benefits:**
- ✅ Detects production issues (agents stuck in loops)
- ✅ Validates error handling logic
- ✅ Important for reliability

**Not in DeepEval**: This specific metric doesn't exist in DeepEval. Should be added to UAEF!

**Dimension**: Tool Calling (complements existing tool metrics)

**Updated Tool Calling Metrics (4 → 5):**
- tool_selection_accuracy
- tool_sequence_correctness  
- parameter_quality
- mcp_compliance
- **error_recovery_quality** ← NEW

---

### Recommendation: Leverage DeepEval ⭐

**Instead of reimplementing**, use UAEF's existing DeepEval integration:

```python
from uaef.api import evaluate
from uaef.integrations import DeepEvalConnector

# Standard UAEF metrics
result = evaluate(trace=trace, ground_truth=ground_truth, adapter="langgraph")

# Add DeepEval metrics for planning agents
deepeval = DeepEvalConnector()
if deepeval.is_available():
    # Plan quality (requires trace with plan)
    plan_quality = deepeval.calculate_plan_quality(eval_input)
    
    # Plan adherence (requires trace with plan)
    plan_adherence = deepeval.calculate_plan_adherence(eval_input)
    
    # Step efficiency (execution path analysis)
    step_efficiency = deepeval.calculate_step_efficiency(eval_input)
    
    # Add to results
    result.add_external_metrics([plan_quality, plan_adherence, step_efficiency])
```

**Benefits:**
- ✅ No implementation needed (already available)
- ✅ Maintained by DeepEval team
- ✅ Works through UAEF's connector
- ✅ Consistent with UAEF's philosophy of external integrations

### When to Use DeepEval Metrics

| UAEF Metrics | For | DeepEval Metrics | For |
|-------------|-----|-----------------|-----|
| Tool/response/responsible AI | All agents | Plan quality/adherence | Planning agents (ReAct) |
| Performance efficiency | All agents | Step efficiency | Detecting redundancy |
| Context retention | Conversation quality | Knowledge retention | User fact retention |
| Coherence | Overall flow | Turn relevancy | Per-turn relevance |

**Strategy**: Use UAEF for core metrics, supplement with DeepEval for specialized evaluation needs.

---

## Decisions Required

### Decision 1: Tool Selection & Parameter Quality
Should we adopt DeepEval's per-call matching approach?

- [ ] ✅ Yes, implement with mode parameter (backward compatible)
- [ ] ⏸️  Need more discussion
- [ ] ❌ Not a priority

**Effort**: 1-2 weeks | **Risk**: Low | **Impact**: High

### Decision 2: Edge Case Handling ⭐ Unified Approach
Return None + warning with edge case types for all edge cases?

- [ ] 🎯 **Return None + warning** - no exceptions, trackable, filter in aggregation ⭐ **Recommended**
- [ ] 📋 **Hybrid** (exception for some, None for others)
- [ ] 🔥 **Exception** (like DeepEval) - fail fast but breaks pipeline
- [ ] ⏸️  Need more discussion

**Rationale for None + Warning:**
- Clear semantics: None = "not applicable", 0.0 = "agent failed"
- Pipeline doesn't break (no exceptions)
- Trackable via `edge_case_type` in metadata ("missing_data", "not_applicable", "api_error")
- Flexible: partial results possible
- Accurate: only applicable metrics affect score
- Simplifies code: eliminates redundant ground_truth checks

**Implementation**: All 37 problematic edge cases return `score=None` with `metadata["edge_case_type"]`

**Risk**: Medium | **Impact**: High

### Decision 3: Token/Cost Efficiency Improvements
How should quality score be provided and penalty calculated?

**Quality Score:**
- [ ] 🔄 **Two-phase evaluation** - calculate quality first, then efficiency ⭐ Recommended
- [ ] 📥 **Parameter passing** - pass quality_score to efficiency metrics
- [ ] 📋 **Keep default** (0.5) - document limitation

**Penalty Mode:**
- [ ] 🎯 **Add both modes** - "light" (√) and "heavy" (linear) ⭐ Recommended
- [ ] 💡 **Light only** - replace linear with √
- [ ] 📋 **Keep current** - linear penalty only

**Bug Fix:**
- [ ] ✅ **Fix cost threshold** - change `max(0.01, ...)` to `max(1.0, ...)` (critical!)

**Effort**: 3-5 days | **Risk**: Low | **Impact**: Medium

### Decision 4: Context Retention False Positives
How to handle conversations with no context to retain?

- [ ] 🎯 **Return None** when LLM determines no retention needed ⭐ **Recommended**
- [ ] 📝 **Improve prompt** to ask if retention is applicable first
- [ ] 🔬 **Extract knowledge** like DeepEval (complex but most accurate)
- [ ] ⏸️  Need more discussion

**Rationale for None:**
- Accurate: conversation doesn't require retention → metric not applicable
- No false positives: won't score high for independent questions
- Clean aggregation: filters out genuinely non-applicable cases

**Effort**: 1-3 days | **Risk**: Low | **Impact**: Medium
