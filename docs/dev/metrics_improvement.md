# Metrics Improvement - Quick Overview

**Date**: 2026-04-16  
**Time to Read**: 10 minutes

---

## Executive Summary: All Issues

| # | Issue | Metric(s) | Impact | Solution | Priority |
|---|-------|-----------|--------|----------|----------|
| **1** | **Swapped args → 100%** | Tool Selection, Parameter Quality | ❌ False positives | Per-call matching | 🔴 High |
| **2** | **64 edge cases → 1.0/0.0** | All metrics | ⚠️ Score distortion | Return None | 🔴 High |
| **3** | **quality_score = 0.5** | Token/Cost Efficiency | ❌ Meaningless | Two-phase eval | 🟡 Med |
| **4** | **Cost: max(0.01,...)** | Cost Efficiency | 🐛 Bug | max(1.0,...) | 🔴 Crit |
| **5** | **Flat/harsh formula** | Token/Cost Efficiency | ⚠️ Poor scaling | √ penalty modes | 🟡 Med |
| **6** | **Context false positives** | Context Retention | ❌ High for unrelated Qs | Return None | 🟡 Med |
| **7** | **Requires criteria** | Conversation Completeness | ⚠️ Inflexible | Extract intentions | 🟡 Med |
| **8** | **Ignores achievement** | Turn Efficiency | ⚠️ Incomplete | Add completeness | 🟡 Med |
| **9** | **Single-query limits** | Agent Utilization | ℹ️ Limited meaning | Add batch calc | 🟢 Low |
| **10** | **Binary failure** | Workflow Completion | ❌ No partial credit | Partial eval | 🟡 Med |
| **11** | **Hardcoded ratio** | Coordination Efficiency | ⚠️ One-size-fits-all | Configurable | 🟢 Low |
| **12** | **Missing metrics** | Plan/Error Recovery | ⚠️ Gaps | DeepEval + new | 🟡 Med |

**Stats**: 12 issues, 4 critical/high priority

---

## Issue 1: Tool Selection - Three Problems

### Problem A: Swapped Arguments → 100%
```python
Expected: subtract(a=100, b=25)  # = 75
Called:   subtract(a=25, b=100)  # = -75 ❌
Score: 100% FALSE POSITIVE!
```
**Cause**: Bag-of-values - checks if values exist, not which parameter.

### Problem B: Separated Evaluation
```python
score = 1.0 - (0.4×incorrect_tools + 0.4×missed_tools + 0.2×(1-args))
```
Evaluates **tool names** and **arguments separately**:
- Tool name ✓ | Arguments present ✓ | But **binding wrong** ❌

### Problem C: Fixed Weights
Hardcoded 40%/40%/20% - no flexibility for different use cases.

### Solution: Holistic Per-Call Matching
```python
for each expected_call:
    find best matching actual_call
    compare tool+params together as key-value pairs
score = average match quality
```

**Impact**: ✅ Detects swaps | ✅ Unified evaluation | ✅ Flexible

📖 **See [Appendix 1](#appendix-1-tool-selection---detailed-formula-analysis) for formula breakdown**

---

## Issue 2: Edge Cases Return Scores

### Problem

64 edge cases return **1.0 or 0.0** instead of None:

| Metric | Edge Case | Current | Should |
|--------|-----------|---------|--------|
| mcp_compliance | No tool calls | 1.0 ⚠️ | None |
| token_efficiency | No token data | 1.0 ⚠️ | None |
| answer_relevance | API error | 0.0 ❌ | None |
| context_retention | < 2 turns | 1.0 ⚠️ | None |

**Example Impact:**
```
10 metrics: [0.9, 0.8, 1.0, 1.0, 1.0, 0.0, 0.0, 0.85, 0.75, 0.9]
             Real: [0.9, 0.8, 0.85, 0.75, 0.9]
             Inflated/Deflated: [1.0, 1.0, 1.0, 0.0, 0.0]

Current average: 0.72 (misleading!)
Should be: 0.84 (accurate - only 5 applicable metrics)
```

### Solution

Return **None** when metric not applicable:
```python
if len(tool_calls) == 0:
    return MetricScore(score=None, applicable=False)

# Filter in aggregation
valid_scores = [s for s in scores if s.score is not None]
average = sum(valid_scores) / len(valid_scores)
```

**Note**: Current code has **redundant checking** - checks both `ground_truth` existence and `expected_output`. Should simplify to just check `expected_output` directly.

📖 **See [Appendix 2](#appendix-2-complete-edge-case-list) for all 75 edge cases**

---

## Issue 3: Token/Cost Efficiency

### Problem 1: quality_score Never Set

```python
quality_score = trace.metadata.get("quality_score", 0.5)  # Always 0.5!
```

**All agents get same quality** → efficiency scores meaningless.

### Problem 2: Bad Formula

```python
# Current - issues:
# 1. Flat < 1000 tokens (no reward for efficiency)
# 2. Harsh linear penalty above
# 3. Cost bug: max(0.01, ...) should be max(1.0, ...)

# Proposed:
if tokens <= expected:
    efficiency = quality  # Flat (rewards efficiency)
else:
    efficiency = quality / (ratio ** 0.5)  # Square root penalty (gentler)
```

**Add modes**: "light" (√) and "heavy" (linear) penalty

### Solution

1. **Two-phase evaluation**: Calculate quality metrics first
2. **Fix formula**: Use √ penalty, fix cost bug
3. **Add penalty modes**: User chooses strictness

📖 **See [Appendix 3](#appendix-3-tokencost-efficiency---formula-analysis) for formula details and bug**

---

## Issue 4: Context Retention - False Positives

Independent questions score high even when no context to retain.

**Solution**: LLM returns None if no retention needed.

📖 **See [Appendix 4](#appendix-4-context-retention---deepevals-knowledge-extraction) for DeepEval comparison**

---

## Issue 5: Conversation Completeness - Rigid Criteria

**Problem**: Requires explicit success criteria in ground truth.  
**Solution**: Make optional - extract intentions if not provided (like DeepEval).

---

## Issue 6: Turn Efficiency - Ignores Goal Achievement

**Problem**: 
```python
3 turns, goal achieved (completeness=1.0) → score: 1.0
3 turns, goal failed (completeness=0.3)   → score: 1.0 ❌ SAME!
```

**Solution**: `efficiency = completeness × (expected / actual)` if over threshold.

📖 **See [Appendix 5](#appendix-5-turn-efficiency---completeness-integration) for formula**

---

## Issue 7: Agent Utilization - Single vs Batch

**Problem**: Single-query utilization limited meaning (depends on task - some require more agents some less).  
**Solution**: Add batch calculation, document that batch is more meaningful.

---

## Issue 8: Workflow Completion - Binary Failure

**Problem**: 
```python
if workflow_status == FAILED:
    return 0.0  # No credit for 70% completion!
```

**Solution**: Evaluate partial completion, apply 0.5x penalty for failures.

📖 **See [Appendix 7](#appendix-7-workflow-completion---partial-credit) for scoring table**

---

## Issue 9: Coordination Efficiency - Hardcoded Ratio

**Problem**: `ideal_ratio = 0.1` hardcoded - wrong for sequential pipelines.  
**Solution**: Make configurable per workflow type (0.05 hierarchical, 0.25 sequential).

📖 **See [Appendix 8](#appendix-8-coordination-efficiency---ideal-ratios) for ideal ratios by workflow type**

---

## Issue 10: Missing Metrics

**Use DeepEval integration** (already available):

| Need | DeepEval Has | Type |
|------|-------------|------|
| Plan quality/adherence | ✅ | Planning agents |
| Step efficiency | ✅ | Path analysis |
| Knowledge retention | ✅ | Granular facts |

**New metric to add**:
- `error_recovery_quality` - Detects repeated failed attempts (Tool Calling dimension)

---

## Appendix 1: Tool Selection - Detailed Formula Analysis

**Current formula with swapped args example:**
```python
score = 1.0 - (0.4×incorrect + 0.4×missed + 0.2×(1-args))

subtract(a=25,b=100) vs expected(a=100,b=25):
- incorrect = 0.0  # Tool name correct
- missed = 0.0     # No tools missed  
- args = 1.0       # Values [100,25] present
- score = 1.0 - 0 = 1.0 ❌
```

**DeepEval's key-value approach:**
```python
compare_params(expected, actual):
    {a:100,b:25} vs {a:25,b:100}
    → a mismatch, b mismatch
    → 0/2 = 0.0 ✓
```

---

## Appendix 2: Complete Edge Case List

### 75 Edge Case Handlers Found

| Dimension | Metric | Edge Case | Current | Issue | Proposed |
|-----------|--------|-----------|---------|-------|----------|
| **Tool Calling** | tool_selection_accuracy | Both empty (expected & called) | 1.0 | ✓ Valid | Keep 1.0 |
| | tool_selection_accuracy | Expected empty, called some | 0.0 | ✓ Valid | Keep 0.0 |
| | tool_selection_accuracy | Expected some, called empty | 0.0 | ✓ Valid | Keep 0.0 |
| | tool_selection_accuracy | No ground truth | Exception | ✓ Valid | **None + warning** |
| | tool_sequence_correctness | Both empty | 1.0 | ✓ Valid | Keep 1.0 |
| | tool_sequence_correctness | One empty | 0.0 | ✓ Valid | Keep 0.0 |
| | tool_sequence_correctness | No ground truth | Exception | ✓ Valid | **None + warning** |
| | parameter_quality | Both empty | 1.0 | ✓ Valid | Keep 1.0 |
| | parameter_quality | One empty | 0.0 | ✓ Valid | Keep 0.0 |
| | parameter_quality | No ground truth | Exception | ✓ Valid | **None + warning** |
| | mcp_compliance | No tool calls | 1.0 | ⚠️ Inflation | **None + warning** |
| **Performance** | latency_score | No latency data | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| | latency_score | Negative latency (invalid) | 0.0 | ⚠️ Invalid data | **None + warning** (invalid_data) |
| | token_efficiency | No token data | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| | cost_efficiency | No cost data | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| | cost_efficiency | Negative cost (invalid) | 0.0 | ✓ Valid | Keep 0.0 |
| | throughput | No throughput data | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| | throughput | Negative throughput (invalid) | 0.0 | ✓ Valid | Keep 0.0 |
| **Response Quality** | answer_relevance | No question | 0.0 | ❌ Deflation | ✅ **None + warning** (missing_data) |
| | answer_relevance | Empty question | 0.0 | ❌ Deflation | ✅ **None + warning** (invalid_data) |
| | answer_relevance | No response | 0.0 | ❌ Deflation | ✅ **None + warning** (missing_data) |
| | answer_relevance | Empty response | N/A | N/A | ✅ **0.0 + warning** (invalid_data) |
| | answer_relevance | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | completeness | No ground truth | Exception | ⚠️ Redundant | ✅ **None + warning** (missing_data) |
| | completeness | No response | 0.0 | ❌ Deflation | ✅ **None + warning** (missing_data) |
| | completeness | Empty response | N/A | N/A | ✅ **0.0 + warning** (invalid_data) |
| | completeness | No expected output* | Exception/1.0 | ⚠️ Redundant | ✅ **None + warning** (invalid_data) |
| | completeness | Empty expected output | 1.0 | ⚠️ Inflation | ✅ **None + warning** (invalid_data) |
| | completeness | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | hallucination_score | No response | 1.0 | ⚠️ Inflation | ✅ **None + warning** (missing_data) |
| | hallucination_score | Empty response | 1.0 | ⚠️ Inflation | ✅ **0.0 + warning** (invalid_data) |
| | hallucination_score | No context | 1.0 | ⚠️ Inflation | ✅ **None + warning** (missing_data) |
| | hallucination_score | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | accuracy | No ground truth | Exception | ❌ Redundant | ✅ **None + warning** (missing_data) |
| | accuracy | No question | 0.0 | ❌ Deflation | ✅ **None + warning** (missing_data) |
| | accuracy | Empty question | 0.0 | ❌ Deflation | ✅ **None + warning** (invalid_data) |
| | accuracy | No response | 0.0 | ❌ Deflation | ✅ **None + warning** (missing_data) |
| | accuracy | Empty response | N/A | N/A | ✅ **0.0 + warning** (invalid_data) |
| | accuracy | No expected output* | Exception/0.0 | ❌ Redundant | ✅ **None + warning** (invalid_data) |
| | accuracy | Empty reference answer | 0.0 | ❌ Deflation | ✅ **None + warning** (invalid_data) |
| | accuracy | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| **Responsible AI** | safety_score | No response | 1.0 | ⚠️ Inflation | **None + warning** (missing_data) |
| | safety_score | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | bias_score | No response | 1.0 | ⚠️ Inflation | **None + warning** (missing_data) |
| | bias_score | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | toxicity_score | No response | 1.0 | ⚠️ Inflation | **None + warning** (missing_data) |
| | toxicity_score | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | prompt_injection_detection | No user messages | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| **Multi-Turn** | context_retention | < 4 messages (< 2 turns) | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| | context_retention | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | coherence | < 4 messages (< 2 turns) | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| | coherence | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | conversation_completeness | No expected output* | Exception/1.0 | ⚠️ Redundant | **None + warning** (missing_data) |
| | conversation_completeness | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | turn_efficiency | 0 turns | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| **Multi-Agent** | agent_utilization | Not multi-agent | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| | agent_utilization | No agents/work | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| | delegation_quality | Not multi-agent | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| | delegation_quality | No coordination | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| | delegation_quality | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | workflow_completion | Not multi-agent | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| | workflow_completion | No expected output* | Exception/1.0 | ⚠️ Redundant | **None + warning** (missing_data) |
| | workflow_completion | Workflow FAILED status | 0.0 | ✓ Valid | Keep 0.0** |
| | workflow_completion | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | coordination_efficiency | Not multi-agent | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| | coordination_efficiency | No work | 1.0 | ⚠️ Inflation | **None + warning** (not_applicable) |
| **Reasoning** | chain_of_thought_coherence | No reasoning | 1.0 | ⚠️ Inflation | **None + warning** (missing_data) |
| | chain_of_thought_coherence | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | logical_consistency | No reasoning | 1.0 | ⚠️ Inflation | **None + warning** (missing_data) |
| | logical_consistency | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | reasoning_step_correctness | No reasoning | 1.0 | ⚠️ Inflation | **None + warning** (missing_data) |
| | reasoning_step_correctness | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |
| | fallacy_detection | No reasoning | 1.0 | ⚠️ Inflation | **None + warning** (missing_data) |
| | fallacy_detection | API error | 0.0 | ❌ Deflation | **None + warning** (api_error) |

**Summary**: 
- **Inflation** (returns 1.0 when not applicable): 19 cases ⚠️ → **Return None**
- **Deflation** (returns 0.0 when can't evaluate): 16 cases ❌ → **Return None**
- **Invalid data** (negative/invalid values): 3 cases ⚠️ → **Return None**
- **Valid** (correct behavior): 24 cases ✓ → **Keep as-is**
- **New** (empty input edge cases, not previously handled): 5 cases → **0.0 or None + warning**
- ✅ **Implemented** (Response Quality): 19 cases fixed
- **Total**: 75 edge case handlers
- **Remaining**: 21 (non-Response Quality dimensions)

**Notes**:
- *Redundant: Checks both `ground_truth` and `expected_output`. Simplify to only check `expected_output`.
- **Workflow FAILED should evaluate partial completion and apply penalty, not return 0.0

**Unified Approach**: All problematic cases **return None with warning in metadata**

```python
# Simplify redundant checks
expected_output = (evaluation_input.ground_truth.expected_output 
                   if evaluation_input.ground_truth else None)

if not expected_output:
    return MetricScore(
        score=None,
        applicable=False,
        reasoning="Cannot evaluate: no expected output",
        metadata={"warning": "edge_case", "edge_case_type": "missing_data"}
    )

# API errors
except ClientError as e:
    return MetricScore(
        score=None,
        applicable=False,
        reasoning=f"Cannot evaluate: API error: {e}",
        metadata={"warning": "edge_case", "edge_case_type": "api_error"}
    )
```

**Benefits**:
- ✅ Pipeline doesn't break (no exceptions)
- ✅ Clear warnings in metadata
- ✅ Trackable edge case types
- ✅ Filters out in aggregation
- ✅ Simpler code (no redundant checks)

**Decision Rule**:
- Missing data / API error / Not applicable / Invalid data → **None + warning**
- Actual failure → **Keep low score** (legitimate failure)

---

## Appendix 3: Token/Cost Efficiency - Formula Analysis

**Flat region issue:**
```
100 tokens  → 0.8
500 tokens  → 0.8  
999 tokens  → 0.8  ← No reward!
```

**Cost efficiency bug:**
```python
# WRONG: max(0.01,...)
$0.0001 → 80.0 → clamped to 1.0

# CORRECT: max(1.0,...)  
$0.0001 → 0.8 ✓
```

---

**Proposed comparison:**
```
Current vs Light (√) penalty:
2000 tokens: 0.40 → 0.57 (42% less harsh)
5000 tokens: 0.16 → 0.36 (125% less harsh)
```

---

## Appendix 4: Context Retention - DeepEval's Knowledge Extraction

**DeepEval's Approach:**

**Step 1**: Extract knowledge from user messages
```
User: "I live in Paris" → {"Location": "Paris"}
User: "What's 2+2?"     → {} (no user facts)
```

**Step 2**: Check retention in each assistant response  
**Step 3**: Score = retention_count / verdicts (or 0 if no verdicts)

**Key**: Returns 0 when nothing to retain, not high score!

---

## Appendix 5: Turn Efficiency - Completeness Integration

**Current (ignores completeness):**
```
3 turns, complete → 1.0
3 turns, incomplete → 1.0 ❌
```

**Proposed:**
```
efficiency = completeness × (threshold_factor)
3 turns, completeness=1.0 → 1.0
3 turns, completeness=0.3 → 0.3 ✓
```

---

---

## Appendix 7: Workflow Completion - Partial Credit

| Completion | Status | Current | Proposed |
|------------|--------|---------|----------|
| 70% | FAILED | 0.0 | **0.35** (70%×0.5) |
| 30% | FAILED | 0.0 | **0.15** (30%×0.5) |
| 100% | COMPLETED | 1.0 | 1.0 |

---

## Appendix 8: Coordination Efficiency - Ideal Ratios

| Workflow Type | Ideal Ratio | Why |
|---------------|-------------|-----|
| Hierarchical | 0.05 | Centralized orchestrator |
| Balanced | 0.10 | Default |
| Collaborative | 0.20 | Peer-to-peer |
| Sequential | 0.25 | Many handoffs |

**Current (hardcoded 0.1)**: Penalizes collaborative (0.20) and sequential (0.25) workflows.

---

**Questions?** See [full technical doc](./metrics_improvement_details.md) for complete details.
