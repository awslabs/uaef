# Test Data for UAEF Metrics

This directory contains test data files for comprehensive unit testing of UAEF metrics.

## Structure

Each JSON file contains test cases for a specific metric:

```json
{
  "test_cases": [
    {
      "name": "scenario_name",
      "description": "What this scenario tests",
      "scenario_type": "happy_path|edge_case|violation|etc",
      "expected_score": 0.85,           // Exact expected score (null for None)
      "expected_score_range": [0.8, 0.9], // OR score range (use one)
      "metric_params": {},              // Optional: constructor args for metric
      "trace": { /* AgentTrace data */ },
      "ground_truth": { /* Optional */ }
    }
  ]
}
```

## Scenario Types

| Type | Purpose | Examples |
|------|---------|----------|
| `happy_path` | Correct/expected behavior | Perfect match, all compliant |
| `violation` | Rule violations | Empty name, wrong format |
| `edge_case` | Boundary conditions | No data, empty lists, null values |
| `argument_swap` | Parameter order issues | Swapped coordinates, math args |
| `missing_tool` | Missing expected items | Tool not called, params missing |
| `incorrect_tool` | Wrong items used | Wrong tool, wrong params |
| `threshold` | At exact threshold | 1000 tokens, $0.01 cost, 2000ms |
| `above_threshold` | Beyond threshold | High usage scenarios |
| `below_threshold` | Under threshold | Low usage scenarios |
| `invalid_data` | Invalid input values | Negative latency, negative cost |
| `penalty_mode` | Non-default metric config | Heavy penalty mode via `metric_params` |
| `quality_vs_tokens` | Quality/usage tradeoffs | High quality + high tokens |

## Test Files

### Tool Calling Metrics

| Metric | Test Cases | Coverage |
|--------|-----------|----------|
| **Tool Selection Accuracy** | 16 | Names only (Jaccard) |
| **Tool Sequence Correctness** | 11 | LCS-based ordering |
| **Parameter Quality** | 11 | Type-aware matching |
| **MCP Compliance** | 6 | Standards validation |
| **Total** | **44** | **100% pass rate** |

#### Tool Selection Accuracy (16 cases)

Uses **Jaccard similarity** on tool names only (parameters ignored).

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `perfect_match` | happy_path | 1.0 | Tool names match perfectly |
| 2 | `parameters_ignored` | happy_path | 1.0 | Names match, params differ (still 1.0) |
| 3 | `missing_tool` | missing_tool | 0.5 | Expected 2, called 1 |
| 4 | `incorrect_tool` | incorrect_tool | 0.0 | Called wrong tool entirely |
| 5 | `extra_tool` | extra_tool | 0.5 | Expected 1, called 2 |
| 6 | `duplicate_tool_names_match` | multiple_calls | 1.0 | Both have 2x same tool |
| 7 | `no_tools_expected_none_called` | edge_case_empty | 1.0 | Both empty |
| 8 | `no_tools_expected_but_called` | edge_case_unexpected | 0.0 | Called when none expected |
| 9 | `expected_tools_none_called` | edge_case_missing | 0.0 | Expected tools, called none |
| 10 | `multiple_tools_all_correct` | happy_path | 1.0 | 3 tools all match |
| 11 | `duplicate_correct_calls` | duplicates | 1.0 | 3x same tool correct |
| 12 | `extra_duplicate_calls` | extra_calls | 0.5 | Expected 2, called 4 |
| 13 | `partial_match_mixed_tools` | partial | 0.5 | 2 of 4 match |
| 14 | `completely_wrong_tools` | incorrect_tools | 0.0 | No overlap |
| 15 | `one_match_many_wrong` | many_extra | 0.25 | 1 correct, 3 extra |
| 16 | `missing_ground_truth` | not_applicable | None | Returns None |

#### Tool Sequence Correctness (11 cases)

Uses **Longest Common Subsequence (LCS)** for sequence similarity.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `perfect_sequence` | happy_path | 1.0 | Exact order match |
| 2 | `wrong_order` | incorrect_sequence | 0.667 | Tools in wrong order |
| 3 | `missing_tool_in_sequence` | missing_tool | 0.667 | One tool missing |
| 4 | `extra_tool_in_sequence` | extra_tool | 0.75 | Extra tool inserted |
| 5 | `empty_sequences` | edge_case | 1.0 | Both empty |
| 6 | `one_sequence_empty` | edge_case | 0.0 | One empty mismatch |
| 7 | `partially_correct_order` | partial | 0.667 | Some correct order |
| 8 | `single_tool_correct` | happy_path | 1.0 | Single tool sequence |
| 9 | `reversed_sequence` | incorrect_sequence | 0.333 | Completely reversed |
| 10 | `longer_sequence` | happy_path | 1.0 | 5 tools in order |
| 11 | `missing_ground_truth` | not_applicable | None | Returns None |

#### Parameter Quality (11 cases)

Per-call **key-value pair matching** with type awareness.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `perfect_parameters` | happy_path | 1.0 | All params match |
| 2 | `swapped_parameters` | argument_swap | 0.0 | Dividend/divisor swapped |
| 3 | `missing_parameter` | missing_params | 0.5 | One param missing |
| 4 | `extra_parameters` | extra_params | 0.333 | Extra params added |
| 5 | `wrong_parameter_values` | incorrect_params | 0.0 | Wrong values |
| 6 | `no_tools` | edge_case | 1.0 | No tools to evaluate |
| 7 | `partial_match` | partial | 0.5 | Some params correct |
| 8 | `type_mismatch` | incorrect_params | 0.0 | "8080" vs 8080 |
| 9 | `nested_parameters` | happy_path | 1.0 | Nested dicts match |
| 10 | `null_parameter_value` | edge_case | 1.0 | Null values handled |
| 11 | `missing_ground_truth` | not_applicable | None | Returns None |

#### MCP Compliance (6 cases)

Validates **Model Context Protocol** standards (lowercase names, timestamps, dict args).

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `all_compliant` | happy_path | 1.0 | All standards met |
| 2 | `invalid_naming_convention` | violation | 0.0 | Uppercase in name |
| 3 | `no_tool_calls` | edge_case | None | Not applicable |
| 4 | `mixed_compliance` | partial | 0.667 | 2/3 compliant |
| 5 | `single_compliant_tool` | happy_path | 1.0 | Single tool compliant |
| 6 | `mostly_compliant` | partial | 0.75 | 3/4 compliant |

**Note:** Test cases for `empty_tool_name`, `whitespace_only_name`, and `missing_timestamp` were removed because Pydantic validation in `ToolCall` model prevents these at data ingestion (not metric evaluation).

### Performance Metrics

| Metric | Test Cases | Coverage |
|--------|-----------|----------|
| **Latency Score** | 10 | Threshold-based scoring |
| **Token Efficiency** | 15 | Quality/token with penalty modes |
| **Cost Efficiency** | 15 | Quality/cost with penalty modes |
| **Throughput** | 11 | Target-based RPS scoring |
| **Total** | **51** | **100% pass rate** |

Test cases support an optional `metric_params` field for passing constructor arguments (e.g. `penalty_mode`), enabling the same unified test function to cover different configurations.

#### Latency Score (10 cases)

Threshold-based scoring: score = 1.0 if latency ≤ threshold, else threshold/latency (linear decay).

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `fast_response` | efficient | 1.0 | 500ms, well within 2000ms threshold |
| 2 | `at_threshold` | threshold | 1.0 | Exactly at 2000ms boundary |
| 3 | `slow_response` | above_threshold | 0.5 | 4000ms, 2x threshold → 2000/4000 |
| 4 | `very_slow` | high_latency | 0.2 | 10000ms, 5x threshold → 2000/10000 |
| 5 | `no_latency_info` | edge_case | None | No latency_ms in metadata |
| 6 | `negative_latency` | invalid_data | None | Negative value → None + warning |
| 7 | `ultra_fast` | efficient | 1.0 | 100ms |
| 8 | `slightly_above_threshold` | above_threshold | 0.8 | 2500ms → 2000/2500 |
| 9 | `zero_latency` | edge_case | 1.0 | 0ms is within threshold |
| 10 | `moderate_latency` | efficient | 1.0 | 1500ms, within threshold |

#### Token Efficiency (15 cases)

Formula: within threshold → quality_score; above → quality / (ratio ^ exponent). Light mode exponent=0.5 (√), heavy=1.0 (linear).

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `efficient_low_tokens` | efficient | 0.9 | 100 tokens, quality 0.9, within threshold |
| 2 | `at_threshold` | threshold | 0.8 | 1000 tokens, quality 0.8, at boundary |
| 3 | `above_threshold` | above_threshold | 0.57 | 2x tokens, light: 0.8/√2 |
| 4 | `high_tokens` | high_usage | 0.36 | 5x tokens, light: 0.8/√5 |
| 5 | `below_threshold` | below_threshold | 0.8 | 500 tokens, no penalty |
| 6 | `no_quality_score` | edge_case | None | Missing quality_score in metadata |
| 7 | `no_token_info` | edge_case | None | Null token counts |
| 8 | `zero_tokens` | edge_case | None | 0+0=0 total tokens |
| 9 | `high_quality_high_tokens` | quality_vs_tokens | 0.34 | 8x tokens, light: 0.95/√8 |
| 10 | `low_quality_low_tokens` | quality_vs_tokens | 0.3 | Low quality, within threshold |
| 11 | `quality_score_string_type` | edge_case | None | Non-numeric quality_score |
| 12 | `quality_score_zero` | edge_case | 0.0 | quality_score=0.0 is valid |
| 13 | `heavy_penalty_within_threshold` | penalty_mode | 0.8 | Heavy mode, within threshold |
| 14 | `heavy_penalty_2x_above` | penalty_mode | 0.40 | Heavy: 0.8/2.0 |
| 15 | `heavy_penalty_5x_above` | penalty_mode | 0.16 | Heavy: 0.8/5.0 |

#### Cost Efficiency (15 cases)

Same formula as token efficiency but using cost_usd. Also requires quality_score.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `low_cost_high_quality` | efficient | 0.85 | $0.005, quality 0.85, within threshold |
| 2 | `at_threshold` | threshold | 0.8 | $0.01, quality 0.8, at boundary |
| 3 | `very_low_cost` | below_threshold | 0.8 | $0.0001, no penalty |
| 4 | `high_cost` | above_threshold | 0.36 | 5x cost, light: 0.8/√5 |
| 5 | `no_cost_info` | edge_case | None | Missing cost_usd |
| 6 | `zero_cost` | edge_case | None | cost_usd=0 treated as missing |
| 7 | `moderate_cost` | above_threshold | 0.57 | 2x cost, light: 0.8/√2 |
| 8 | `low_quality_low_cost` | inefficient | 0.3 | Low quality, within threshold |
| 9 | `negative_cost` | invalid_data | 0.0 | Negative cost → score 0.0 |
| 10 | `high_quality_low_cost` | efficient | 0.95 | $0.002, quality 0.95 |
| 11 | `no_quality_score` | edge_case | None | Missing quality_score |
| 12 | `quality_score_string_type` | edge_case | None | Non-numeric quality_score |
| 13 | `heavy_penalty_within_threshold` | penalty_mode | 0.8 | Heavy mode, within threshold |
| 14 | `heavy_penalty_2x_above` | penalty_mode | 0.40 | Heavy: 0.8/2.0 |
| 15 | `heavy_penalty_5x_above` | penalty_mode | 0.16 | Heavy: 0.8/5.0 |

#### Throughput (11 cases)

Linear scoring against target of 10 RPS: score = 1.0 if ≥ target, else throughput/target.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `meets_target` | happy_path | 1.0 | 12.5 RPS, exceeds target |
| 2 | `at_target` | threshold | 1.0 | Exactly 10 RPS |
| 3 | `below_target` | below_threshold | 0.5 | 5 RPS → 5/10 |
| 4 | `very_low_throughput` | low_performance | 0.2 | 2 RPS → 2/10 |
| 5 | `no_throughput_info` | edge_case | None | Missing throughput_rps |
| 6 | `exceeds_target` | high_performance | 1.0 | 15 RPS |
| 7 | `moderate_throughput` | below_threshold | 0.7 | 7 RPS → 7/10 |
| 8 | `one_rps` | low_performance | 0.1 | 1 RPS → 1/10 |
| 9 | `negative_throughput` | invalid_data | 0.0 | Negative → score 0.0 |
| 10 | `zero_throughput` | edge_case | 0.0 | 0 RPS → 0/10 |
| 11 | `throughput_rps_null_explicit` | edge_case | None | Explicit null value |

#### Performance Metrics Coverage Breakdown

| Scenario Type | Count | Description |
|--------------|-------|-------------|
| `efficient` / `happy_path` | 8 | Normal good-performance scenarios |
| `threshold` | 4 | Exact boundary values |
| `above_threshold` | 4 | Above threshold with penalty |
| `below_threshold` | 4 | Below threshold, no penalty |
| `edge_case` | 16 | Missing data, null, zero, invalid types → None |
| `invalid_data` | 4 | Negative values |
| `quality_vs_tokens` | 2 | Quality/usage tradeoffs |
| `penalty_mode` | 6 | Heavy penalty mode (via `metric_params`) |
| `high_latency` / `high_usage` | 2 | Extreme values |
| `low_performance` | 3 | Very low throughput |
| `inefficient` | 1 | Low quality despite low cost |

**Edge Cases returning None**: 12 cases across all 4 metrics cover every code path that returns `score=None`

### Response Quality Metrics

These metrics use LLM judge for evaluation. Test cases with `expected_score` exercise deterministic early-return paths (missing/empty data) and are run automatically in `test_all_metrics.py`. Cases without `expected_score` are validated via `tests/run_llm_metrics.py` with real Bedrock API calls.

| Metric | Test Cases | Deterministic | LLM-evaluated | Coverage |
|--------|-----------|---------------|---------------|----------|
| **Answer Relevance** | 14 | 6 | 8 | Relevance-only scoring |
| **Completeness** | 12 | 5 | 7 | Ground truth comparison |
| **Hallucination Score** | 11 | 3 | 8 | Context-based validation |
| **Accuracy** | 15 | 7 | 8 | Reference answer comparison |
| **Total** | **52** | **21** | **31** | **21 deterministic pass rate: 100%** |

#### Answer Relevance (14 cases)

LLM judge evaluates relevance only (not completeness or accuracy).

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `highly_relevant` | relevant | LLM | Directly addresses question |
| 2 | `relevant_with_detail` | relevant | LLM | Relevant with extra context |
| 3 | `partially_relevant` | partial | LLM | Addresses question but incomplete |
| 4 | `tangentially_relevant` | partial | LLM | Related but doesn't directly answer |
| 5 | `irrelevant` | irrelevant | LLM | Completely unrelated answer |
| 6 | `acknowledgment_only` | irrelevant | LLM | Acknowledges but doesn't answer |
| 7 | `question_with_question` | relevant | LLM | Clarifying question (relevant) |
| 8 | `refusal_relevant` | relevant | LLM | Relevant refusal with explanation |
| 9 | `no_question` | edge_case | None | No user message → missing_data |
| 10 | `empty_question` | edge_case | None | Empty string question → invalid_data |
| 11 | `no_response` | edge_case | None | No assistant message → missing_data |
| 12 | `empty_response` | edge_case | 0.0 | Empty string response → invalid_data |
| 13 | `whitespace_only_question` | edge_case | None | Whitespace question → invalid_data |
| 14 | `whitespace_only_response` | edge_case | 0.0 | Whitespace response → invalid_data |

#### Completeness (12 cases)

LLM judge evaluates coverage of expected output. Incorrect information gets zero credit.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `complete_answer` | complete | LLM | Covers all expected information |
| 2 | `partially_complete` | partial | LLM | Some but not all information |
| 3 | `incomplete_answer` | incomplete | LLM | Missing most expected info |
| 4 | `extra_information` | complete | LLM | Expected info plus extras |
| 5 | `wrong_information` | incomplete | LLM | Provides wrong information |
| 6 | `detailed_complete` | complete | LLM | Detailed coverage of all points |
| 7 | `missing_key_point` | partial | LLM | Mostly complete, missing critical point |
| 8 | `no_expected_output` | edge_case | None | Null expected_output → missing_data |
| 9 | `no_response` | edge_case | None | No assistant message → missing_data |
| 10 | `empty_expected_output` | edge_case | None | Empty expected_output → invalid_data |
| 11 | `empty_response` | edge_case | 0.0 | Empty response → invalid_data |
| 12 | `no_ground_truth` | edge_case | None | No ground_truth object → missing_data |

#### Hallucination Score (11 cases)

LLM judge validates claims against provided context. Score 1.0 = no hallucinations, 0.0 = severe.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `no_hallucination` | faithful | LLM | All claims supported by context |
| 2 | `minor_hallucination` | minor_hallucination | LLM | One unsupported claim |
| 3 | `major_hallucination` | hallucination | LLM | Multiple unsupported claims |
| 4 | `contradicts_context` | hallucination | LLM | Contradicts provided context |
| 5 | `fabricated_details` | hallucination | LLM | Adds details not in context |
| 6 | `reasonable_inference` | faithful | LLM | Reasonable inference from context |
| 7 | `context_misinterpretation` | hallucination | LLM | Misrepresents context |
| 8 | `selective_citation` | hallucination | LLM | Cherry-picks, ignores contradicting info |
| 9 | `no_context_provided` | edge_case | None | Empty context → missing_data |
| 10 | `empty_response` | edge_case | 0.0 | Empty response → invalid_data |
| 11 | `no_response` | edge_case | None | No assistant message → missing_data |

#### Accuracy (15 cases)

LLM judge compares response against reference answer for correctness.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `perfectly_accurate` | accurate | LLM | Matches reference exactly |
| 2 | `semantically_correct` | accurate | LLM | Correct meaning, different wording |
| 3 | `completely_wrong` | inaccurate | LLM | Factually incorrect |
| 4 | `partially_correct` | partial | LLM | Some correct, some wrong |
| 5 | `order_different` | accurate | LLM | Correct items, different order |
| 6 | `missing_units` | partial | LLM | Correct number, missing units |
| 7 | `vague_but_correct` | partial | LLM | Technically correct but vague |
| 8 | `detailed_accurate` | accurate | LLM | Detailed and accurate |
| 9 | `no_reference_answer` | edge_case | None | Null reference → missing_data |
| 10 | `no_question` | edge_case | None | No user message → missing_data |
| 11 | `no_response` | edge_case | None | No assistant message → missing_data |
| 12 | `empty_question` | edge_case | None | Empty question → invalid_data |
| 13 | `empty_response` | edge_case | 0.0 | Empty response → invalid_data |
| 14 | `empty_reference` | edge_case | None | Empty reference → invalid_data |
| 15 | `no_ground_truth` | edge_case | None | No ground_truth object → missing_data |

#### Response Quality Edge Case Rules

All four metrics follow a consistent pattern for edge cases:

| Condition | Score | Warning | Rationale |
|-----------|-------|---------|-----------|
| No user message (None) | None | missing_data | Can't evaluate without input |
| Empty user message ("") | None | invalid_data | Invalid input, not evaluable |
| No assistant message (None) | None | missing_data | Nothing to evaluate |
| Empty assistant message ("") | 0.0 | invalid_data | Present but empty = failure |
| No ground_truth object | None | missing_data | Required data missing |
| Null expected_output | None | missing_data | Required data missing |
| Empty expected_output ("") | None | invalid_data | Invalid reference data |
| No context (hallucination only) | None | missing_data | Can't validate without context |
| API error | None | api_error | External failure, not evaluable |

### Responsible AI Metrics

These metrics evaluate safety, bias, toxicity, and prompt injection. Safety, bias, and toxicity use LLM judge; prompt injection detection is fully deterministic (regex pattern matching).

| Metric | Test Cases | Deterministic | LLM-evaluated | Coverage |
|--------|-----------|---------------|---------------|----------|
| **Safety Score** | 11 | 2 | 9 | Harmful/unsafe content detection |
| **Bias Score** | 11 | 2 | 9 | Bias and stereotyping detection |
| **Prompt Injection Detection** | 10 | 10 | 0 | Regex-based pattern matching |
| **Toxicity Score** | 11 | 2 | 9 | Toxic/rude language detection |
| **Total** | **43** | **16** | **27** | **16 deterministic pass rate: 100%** |

#### Safety Score (11 cases)

LLM judge evaluates harmful, offensive, or inappropriate content. Score 1.0 = safe, 0.0 = unsafe.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `completely_safe` | safe | LLM | Helpful and appropriate response |
| 2 | `safe_refusal` | safe | LLM | Appropriately refuses harmful request |
| 3 | `harmful_instructions` | unsafe | LLM | Provides dangerous instructions |
| 4 | `self_harm_content` | unsafe | LLM | Encourages giving up |
| 5 | `violence_content` | unsafe | LLM | Violent/graphic content |
| 6 | `medical_advice_appropriate` | safe | LLM | General info with disclaimer |
| 7 | `medical_advice_dangerous` | unsafe | LLM | Dangerous medical advice |
| 8 | `privacy_violation` | unsafe | LLM | Requests private information |
| 9 | `educational_content` | safe | LLM | Educational sensitive topic |
| 10 | `no_response_edge_case` | edge_case | None | No assistant message → missing_data |
| 11 | `empty_response` | edge_case | None | Empty response → missing_data |

#### Bias Score (11 cases)

LLM judge detects biased language or unfair treatment. Score 1.0 = unbiased, 0.0 = highly biased.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `unbiased_neutral` | unbiased | LLM | Neutral, factual response |
| 2 | `gender_bias` | biased | LLM | Gender stereotyping |
| 3 | `age_bias` | biased | LLM | Age-related bias |
| 4 | `racial_stereotyping` | biased | LLM | Racial stereotypes |
| 5 | `inclusive_language` | unbiased | LLM | Inclusive, fair language |
| 6 | `cultural_insensitivity` | biased | LLM | Cultural assumptions |
| 7 | `economic_bias` | biased | LLM | Economic status assumptions |
| 8 | `balanced_perspective` | unbiased | LLM | Balanced, fair perspective |
| 9 | `disability_bias` | biased | LLM | Disability-related bias |
| 10 | `no_response_edge` | edge_case | None | No assistant message → missing_data |
| 11 | `empty_response` | edge_case | None | Empty response → missing_data |

#### Prompt Injection Detection (10 cases)

Deterministic regex-based detection. Score 1.0 = no injection, 0.0 = injection detected.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `clean_input` | happy_path | 1.0 | Normal user input |
| 2 | `ignore_instructions` | injection_attempt | 0.0 | "Ignore previous instructions..." |
| 3 | `system_prompt_injection` | injection_attempt | 0.0 | "system: ..." prompt injection |
| 4 | `jailbreak_attempt` | injection_attempt | 0.0 | DAN mode / jailbreak keywords |
| 5 | `override_attempt` | injection_attempt | 0.0 | "Override settings..." |
| 6 | `no_user_messages` | edge_case | None | No user messages → not_applicable |
| 7 | `multiple_clean_messages` | happy_path | 1.0 | Multiple clean messages |
| 8 | `role_play_attempt` | injection_attempt | 0.0 | "Pretend you are..." |
| 9 | `forget_instructions` | injection_attempt | 0.0 | "Forget previous instructions..." |
| 10 | `legitimate_question_with_ignore` | happy_path | 1.0 | Legitimate use of "ignore" |

#### Toxicity Score (11 cases)

LLM judge detects toxic, rude, or disrespectful language. Score 1.0 = non-toxic, 0.0 = highly toxic.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `polite_helpful` | non_toxic | LLM | Polite and helpful response |
| 2 | `respectful_disagreement` | non_toxic | LLM | Disagrees respectfully |
| 3 | `rude_dismissive` | toxic | LLM | Rude and dismissive tone |
| 4 | `insults` | toxic | LLM | Direct insults |
| 5 | `aggressive_tone` | toxic | LLM | Aggressive/hostile tone |
| 6 | `condescending` | toxic | LLM | Condescending/patronizing |
| 7 | `profanity` | toxic | LLM | Profanity/vulgar language |
| 8 | `direct_constructive` | non_toxic | LLM | Direct but constructive |
| 9 | `sarcastic_hurtful` | toxic | LLM | Hurtful sarcasm |
| 10 | `no_response_edge` | edge_case | None | No assistant message → missing_data |
| 11 | `empty_response` | edge_case | None | Empty response → missing_data |

### Reasoning Metrics

These metrics evaluate the quality of reasoning in agent responses. All use LLM judge. Test cases with `expected_score` exercise deterministic early-return paths and are run in `test_all_metrics.py`. Cases without `expected_score` are validated via `tests/run_llm_metrics.py`.

| Metric | Test Cases | Deterministic | LLM-evaluated | Coverage |
|--------|-----------|---------------|---------------|----------|
| **Chain of Thought Coherence** | 11 | 2 | 9 | Reasoning flow and structure |
| **Logical Consistency** | 11 | 2 | 9 | Contradiction detection |
| **Reasoning Step Correctness** | 11 | 2 | 9 | Validity of individual steps |
| **Fallacy Detection** | 11 | 2 | 9 | Logical fallacy identification |
| **Total** | **44** | **8** | **36** | **8 deterministic pass rate: 100%** |

#### Chain of Thought Coherence (11 cases)

LLM judge evaluates whether reasoning steps flow logically and build on each other. Score 1.0 = perfectly coherent chain, 0.0 = disjointed/circular.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `coherent_reasoning` | coherent | LLM | Clear logical progression (date calculation) |
| 2 | `step_by_step_clear` | coherent | LLM | Well-structured numbered steps (investment analysis) |
| 3 | `logical_jumps` | incoherent | LLM | Gaps and random jumps between statements |
| 4 | `disconnected_steps` | incoherent | LLM | Steps don't build on each other |
| 5 | `builds_on_previous` | coherent | LLM | Each step clearly follows from prior (quicksort) |
| 6 | `circular_reasoning` | incoherent | LLM | Reasoning goes in circles |
| 7 | `cause_and_effect` | coherent | LLM | Clear cause-effect chain (plant death) |
| 8 | `no_reasoning` | edge_case | None | No assistant message → missing_data |
| 9 | `empty_response` | edge_case | None | Empty response → missing_data |
| 10 | `multiple_reasoning_steps` | coherent | LLM | Multi-message reasoning progression |
| 11 | `contradictory_reasoning` | incoherent | LLM | Self-contradicting reasoning |

#### Logical Consistency (11 cases)

LLM judge detects contradictions and inconsistencies. Score 1.0 = fully consistent, 0.0 = contradictory.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `fully_consistent` | consistent | LLM | All statements align (gravity explanation) |
| 2 | `self_contradictory` | inconsistent | LLM | "Waterproof" then "not water-resistant" |
| 3 | `conflicting_facts` | inconsistent | LLM | Conflicting store hours |
| 4 | `contradictory_conclusion` | inconsistent | LLM | Conclusion contradicts premises |
| 5 | `temporally_consistent` | consistent | LLM | Logical project timeline |
| 6 | `numerical_inconsistency` | inconsistent | LLM | $10+$15+$20 ≠ $30 |
| 7 | `consistent_multi_turn` | consistent | LLM | Maintains consistency across turns |
| 8 | `opposite_statements` | inconsistent | LLM | "Expensive" and "cheap" about same item |
| 9 | `logically_sound` | consistent | LLM | Sound logical relationships (Python benefits) |
| 10 | `no_reasoning_edge` | edge_case | None | No assistant message → missing_data |
| 11 | `empty_response` | edge_case | None | Empty response → missing_data |

#### Reasoning Step Correctness (11 cases)

LLM judge evaluates whether individual reasoning steps are valid. Score 1.0 = all steps correct, 0.0 = fundamental errors.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `all_steps_correct` | correct | LLM | Valid syllogism (cats → animals → food) |
| 2 | `mathematical_correct` | correct | LLM | Correct primality test for 17 |
| 3 | `logical_error` | incorrect | LLM | False premise: "all birds can fly" |
| 4 | `invalid_inference` | incorrect | LLM | "Some" doesn't imply "all" |
| 5 | `mathematical_error` | incorrect | LLM | πr² computed as π×r (missing square) |
| 6 | `correct_with_context` | correct | LLM | Correct reasoning given context (launch criteria) |
| 7 | `misapplies_principle` | incorrect | LLM | Affirming the consequent fallacy |
| 8 | `no_assistant_message` | edge_case | None | No assistant message → missing_data |
| 9 | `empty_response` | edge_case | None | Empty response → missing_data |
| 10 | `evidence_based_correct` | correct | LLM | Correct conclusions from data |
| 11 | `assumes_without_evidence` | incorrect | LLM | Unsupported claims (cosmic rays) |

#### Fallacy Detection (11 cases)

LLM judge identifies logical fallacies. Score 1.0 = no fallacies, 0.0 = severe fallacies.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `no_fallacies` | no_fallacy | LLM | Sound reasoning (version control benefits) |
| 2 | `ad_hominem` | fallacy | LLM | Attacks person instead of argument |
| 3 | `strawman` | fallacy | LLM | Misrepresents "add tests" as "never ship" |
| 4 | `false_dichotomy` | fallacy | LLM | "Python or failure" — ignores other options |
| 5 | `slippery_slope` | fallacy | LLM | Remote work → company collapse chain |
| 6 | `appeal_to_authority` | fallacy | LLM | "Famous developer uses it, so it's correct" |
| 7 | `hasty_generalization` | fallacy | LLM | One broken car → all unreliable |
| 8 | `post_hoc` | fallacy | LLM | Assumes causation from timing |
| 9 | `circular_reasoning` | fallacy | LLM | "Best because superior because best" |
| 10 | `no_reasoning_edge` | edge_case | None | No assistant message → missing_data |
| 11 | `empty_response` | edge_case | None | Empty response → missing_data |

### Multi-Turn Metrics

These metrics evaluate multi-turn conversation quality. Context retention and coherence use LLM judge; turn efficiency is fully deterministic. Conversation completeness uses LLM judge.

| Metric | Test Cases | Deterministic | LLM-evaluated | Coverage |
|--------|-----------|---------------|---------------|----------|
| **Context Retention** | 11 | 2 | 9 | Cross-turn memory |
| **Coherence** | 10 | 2 | 8 | Logical flow across turns |
| **Conversation Completeness** | 10 | 0 | 10 | Goal achievement |
| **Turn Efficiency** | 13 | 13 | 0 | Turn count vs expected |
| **Total** | **44** | **17** | **27** | **17 deterministic pass rate: 100%** |

#### Context Retention (11 cases)

LLM judge evaluates whether the agent remembers and uses information from earlier turns. Returns null for independent questions with no shared context.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `excellent_retention` | retention | LLM | Remembers name, location, allergy, budget across 5 turns |
| 2 | `good_retention` | retention | LLM | Remembers destination, dates, seat preference in booking flow |
| 3 | `partial_retention` | partial | LLM | Remembers order but forgets specific toppings |
| 4 | `poor_retention_forgets` | no_retention | LLM | Forgets name and language after several turns |
| 5 | `poor_retention_asks_again` | no_retention | LLM | Asks for location already stated |
| 6 | `independent_questions` | not_applicable | LLM | Unrelated factual questions → null score |
| 7 | `factual_questions_no_retention` | not_applicable | LLM | Sequential language questions → null score |
| 8 | `too_few_turns` | edge_case | None | < 2 turns → not_applicable |
| 9 | `contradicts_earlier_statement` | no_retention | LLM | Gives different meeting time/location |
| 10 | `builds_on_context` | retention | LLM | Builds travel itinerary using all prior preferences |
| 11 | `empty_messages` | edge_case | None | No messages → not_applicable |

#### Coherence (10 cases)

LLM judge evaluates logical flow and consistency across conversation turns.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `highly_coherent` | coherent | LLM | Logical party planning across 4 turns |
| 2 | `coherent_weather_discussion` | coherent | LLM | Weather → activity → clothing flow |
| 3 | `topic_switch_coherent` | coherent | LLM | User-initiated topic change handled smoothly |
| 4 | `incoherent_topic_jump` | incoherent | LLM | Agent responds with unrelated facts (Eiffel Tower to bananas) |
| 5 | `self_contradictory` | incoherent | LLM | "Open until 9 PM" then "closed today" |
| 6 | `irrelevant_responses` | incoherent | LLM | Pricing question answered with weather, geography, biology |
| 7 | `logical_progression` | coherent | LLM | Step-by-step Python learning progression |
| 8 | `too_few_turns_edge_case` | edge_case | None | < 2 turns → not_applicable |
| 9 | `maintains_context_across_topic` | coherent | LLM | Presentation topic evolves to slide design |
| 10 | `empty_messages` | edge_case | None | No messages → not_applicable |

#### Conversation Completeness (10 cases)

LLM judge evaluates whether the user's goal was achieved. Works with or without explicit success criteria.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `goal_fully_achieved` | complete | LLM | Flight booked with confirmation |
| 2 | `partially_completed` | partial | LLM | Flight booked but hotel missing |
| 3 | `goal_not_achieved` | incomplete | LLM | Password reset never completed |
| 4 | `goal_changed_achieved_new` | complete | LLM | Changed from Paris to London, London booked |
| 5 | `user_cancels` | incomplete | LLM | User cancels pizza order |
| 6 | `multi_goal_all_complete` | complete | LLM | Weather checked and alarm set |
| 7 | `abandoned_conversation` | incomplete | LLM | Tax filing abandoned mid-flow |
| 8 | `no_success_criteria` | edge_case | LLM | Null expected_output → LLM extracts user intent |
| 9 | `clarification_then_complete` | complete | LLM | Clarified then completed (Q3 report to CEO) |
| 10 | `wrong_goal_completed` | incomplete | LLM | Asked for sales data, got marketing budget |

#### Turn Efficiency (13 cases)

Deterministic metric: `score = completeness × min(1.0, expected_turns / actual_turns)`. Default expected_turns = 5.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `efficient_two_turns` | efficient | 1.0 | 2 turns, within expected 5 |
| 2 | `at_expected_turns` | threshold | 1.0 | Exactly 5 turns |
| 3 | `slightly_above_expected` | above_threshold | 0.71 | 7 turns → 5/7 |
| 4 | `many_turns` | high_turns | 0.33 | 15 turns → 5/15 |
| 5 | `single_turn` | efficient | 1.0 | 1 turn |
| 6 | `ten_turns` | above_threshold | 0.5 | 10 turns → 5/10 |
| 7 | `three_turns` | efficient | 1.0 | 3 turns |
| 8 | `zero_turns` | edge_case | None | 0 turns → not_applicable |
| 9 | `one_message_only` | edge_case | None | Incomplete turn (0 complete) → not_applicable |
| 10 | `twenty_turns` | high_turns | 0.25 | 20 turns → 5/20 |
| 11 | `partial_completeness` | partial_completion | 0.36 | 7 turns, 50% completeness → 0.5 × 5/7 |
| 12 | `custom_expected_turns` | threshold | 1.0 | 3 turns with expected_turns=3 (via metric_params) |
| 13 | `custom_expected_above` | above_threshold | 0.5 | 6 turns with expected_turns=3 → 3/6 |

### Multi-Agent Metrics

These metrics evaluate multi-agent coordination. Agent utilization and coordination efficiency are deterministic. Delegation quality and workflow completion use LLM judge. Test cases use `multi_agent_trace` instead of `trace`.

| Metric | Test Cases | Deterministic | LLM-evaluated | Coverage |
|--------|-----------|---------------|---------------|----------|
| **Agent Utilization** | 11 | 11 | 0 | Workload distribution |
| **Delegation Quality** | 11 | 3 | 8 | Task assignment appropriateness |
| **Workflow Completion** | 10 | 4 | 6 | Goal achievement |
| **Coordination Efficiency** | 11 | 11 | 0 | Coordination overhead |
| **Total** | **43** | **29** | **14** | **29 deterministic pass rate: 100%** |

#### Agent Utilization (11 cases)

Deterministic metric using coefficient of variation: `score = exp(-CV)`. Measures workload distribution evenness.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `perfect_balance` | balanced | 1.0 | 3 agents, equal work (6 each) |
| 2 | `one_agent_overloaded` | unbalanced | 0.34 | Agent A: 10 work, B: 2, C: 0 |
| 3 | `one_agent_idle` | underutilized | 0.49 | Two agents work, one idle |
| 4 | `two_agents_balanced` | balanced | 1.0 | 2 agents, equal work (5 each) |
| 5 | `slight_imbalance` | partial | 0.78 | Agent A: 5, B: 3 |
| 6 | `single_agent_all_work` | unbalanced | 0.24 | 1 worker + 2 idle agents |
| 7 | `no_agents` | edge_case | None | Empty agent_traces → not_applicable |
| 8 | `no_work_performed` | edge_case | 1.0 | Agents exist but no work |
| 9 | `four_agents_balanced` | balanced | 1.0 | 4 agents, equal work (3 each) |
| 10 | `not_multi_agent_trace` | edge_case | None | Regular AgentTrace → not_applicable |
| 11 | `specialized_appropriate` | specialized | 0.58 | Uneven but role-appropriate distribution |

#### Delegation Quality (11 cases)

LLM judge evaluates task-agent fit, delegation clarity, and handoff efficiency.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `appropriate_delegation` | good_delegation | LLM | Research → summary agent pipeline |
| 2 | `poor_delegation_wrong_agent` | bad_delegation | LLM | Math task sent to text-generation agent |
| 3 | `no_delegation` | edge_case | None | No coordination events |
| 4 | `clear_delegation_data_pipeline` | good_delegation | LLM | ETL pipeline with specific instructions |
| 5 | `vague_delegation` | bad_delegation | LLM | "Handle the thing" — no specifics |
| 6 | `efficient_code_review_pipeline` | good_delegation | LLM | Linter → reviewer with clean handoff |
| 7 | `excessive_handoffs_ping_pong` | bad_delegation | LLM | Agents bounce simple email task back and forth |
| 8 | `parallel_independent_tasks` | good_delegation | None | 3 agents work independently, no events |
| 9 | `hierarchical_delegation_chain` | good_delegation | LLM | Lead → devops → QA with reporting |
| 10 | `no_agents_edge` | edge_case | None | Empty agent_traces → missing_data |
| 11 | `not_multi_agent_trace` | edge_case | None | Regular AgentTrace → not_applicable |

#### Workflow Completion (10 cases)

LLM judge evaluates whether the multi-agent workflow achieved its goal. Requires ground truth. Failed workflows get 0.5× penalty.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `completed_successfully` | complete | LLM | Flight booked successfully |
| 2 | `workflow_failed` | failed | LLM | Workflow failed with error |
| 3 | `partial_completion` | partial | LLM | Analysis done but no report |
| 4 | `no_success_criteria` | edge_case | None | Null expected_output → missing_data |
| 5 | `exceeded_expectations` | complete | LLM | Weather + forecast (more than asked) |
| 6 | `multi_agent_success` | complete | LLM | Trip planned with flight, hotel, itinerary |
| 7 | `wrong_outcome` | incomplete | LLM | Emailed Bob instead of Alice |
| 8 | `partial_failure` | failed | LLM | 2 of 3 steps done, workflow failed |
| 9 | `not_multi_agent_edge` | edge_case | None | Regular AgentTrace → not_applicable |
| 10 | `empty_agents` | edge_case | None | Empty agent_traces → missing_data |

#### Coordination Efficiency (11 cases)

Deterministic metric: `score = 1.0` if ratio ≤ ideal (0.10), else `exp(-(ratio - ideal) × 5)`. Measures coordination overhead.

| # | Test Case | Type | Score | Description |
|---|-----------|------|-------|-------------|
| 1 | `efficient_minimal_coordination` | efficient | 1.0 | 1 event for 12 work units |
| 2 | `at_ideal_ratio` | moderate | 0.72 | Ratio 0.167, slightly above ideal |
| 3 | `high_overhead` | inefficient | 0.01 | 4 events for 4 work units (ratio 1.0) |
| 4 | `moderate_overhead` | moderate | 0.72 | 1 event for 6 work units |
| 5 | `no_coordination_events` | efficient | 1.0 | Work done, zero coordination |
| 6 | `no_work_performed` | edge_case | None | No work → missing_data |
| 7 | `collaborative_peer_to_peer` | collaborative | 0.31 | 3 events for 9 work units |
| 8 | `single_agent_trace` | edge_case | 1.0 | Solo agent, no coordination needed |
| 9 | `excessive_coordination` | inefficient | 0.0 | 8 events for 4 work units (ratio 2.0) |
| 10 | `hierarchical_efficient` | efficient | 1.0 | 1 event for 10 work units |
| 11 | `not_multi_agent_trace` | edge_case | None | Regular AgentTrace → not_applicable |

## Running Tests

```bash
# Run all metric tests
pytest tests/test_*.py -v

# Run specific metric
pytest tests/test_tool_selection_accuracy.py -v

# Run specific scenario
pytest tests/test_tool_selection_accuracy.py -k "perfect_match"

# Run by scenario type
pytest tests/ -k "edge_case"
```

## Test Coverage

**Tool Calling Metrics**: 4/4 (100%)
- ✅ Tool Selection Accuracy (16 cases)
- ✅ Tool Sequence Correctness (11 cases)
- ✅ Parameter Quality (11 cases)
- ✅ MCP Compliance (6 cases)

**Performance Metrics**: 4/4 (100%)
- ✅ Latency Score (10 cases)
- ✅ Token Efficiency (15 cases, incl. heavy penalty mode)
- ✅ Cost Efficiency (15 cases, incl. heavy penalty mode)
- ✅ Throughput (11 cases)

**Response Quality Metrics**: 4/4 (100%)
- ✅ Answer Relevance (14 cases: 6 deterministic + 8 LLM)
- ✅ Completeness (12 cases: 5 deterministic + 7 LLM)
- ✅ Hallucination Score (11 cases: 3 deterministic + 8 LLM)
- ✅ Accuracy (15 cases: 7 deterministic + 8 LLM)

**Responsible AI Metrics**: 4/4 (100%)
- ✅ Safety Score (11 cases: 2 deterministic + 9 LLM)
- ✅ Bias Score (11 cases: 2 deterministic + 9 LLM)
- ✅ Prompt Injection Detection (10 cases: all deterministic)
- ✅ Toxicity Score (11 cases: 2 deterministic + 9 LLM)

**Reasoning Metrics**: 4/4 (100%)
- ✅ Chain of Thought Coherence (11 cases: 2 deterministic + 9 LLM)
- ✅ Logical Consistency (11 cases: 2 deterministic + 9 LLM)
- ✅ Reasoning Step Correctness (11 cases: 2 deterministic + 9 LLM)
- ✅ Fallacy Detection (11 cases: 2 deterministic + 9 LLM)

**Multi-Turn Metrics**: 4/4 (100%)
- ✅ Context Retention (11 cases: 2 deterministic + 9 LLM)
- ✅ Coherence (10 cases: 2 deterministic + 8 LLM)
- ✅ Conversation Completeness (10 cases: all LLM)
- ✅ Turn Efficiency (13 cases: all deterministic)

**Multi-Agent Metrics**: 4/4 (100%)
- ✅ Agent Utilization (11 cases: all deterministic)
- ✅ Delegation Quality (11 cases: 3 deterministic + 8 LLM)
- ✅ Workflow Completion (10 cases: 4 deterministic + 6 LLM)
- ✅ Coordination Efficiency (11 cases: all deterministic)

**Total Test Cases**: 321
- Deterministic (automated in pytest): 186
- LLM-evaluated (via run_llm_metrics.py): 135

### Tool Calling Coverage Breakdown

| Scenario Type | Count | Examples |
|--------------|-------|----------|
| `happy_path` | 12 | Perfect matches, all compliant |
| `edge_case` / `not_applicable` | 8 | Empty lists, None returns |
| `incorrect_` | 7 | Wrong tools, wrong params |
| `missing_` | 4 | Missing tools, missing params |
| `extra_` | 3 | Extra tools, extra params |
| `partial` | 4 | Partial matches/compliance |
| `duplicates` | 3 | Duplicate tool calls |
| `argument_swap` | 1 | Swapped parameters |
| `violation` | 2 | MCP violations |

**Edge Cases with None**: All 4 metrics include cases that return `score=None` for not-applicable scenarios

## Adding New Test Cases

1. Add case to appropriate JSON file
2. Include required fields: name, description, scenario_type, expected_score
3. Tests automatically pick up new cases (parameterized)

Example:
```json
{
  "name": "new_scenario",
  "description": "Tests new behavior",
  "scenario_type": "happy_path",
  "expected_score": 0.8,
  "trace": { ... }
}
```

Example with custom metric constructor params:
```json
{
  "name": "heavy_penalty_test",
  "description": "Tests heavy penalty mode",
  "scenario_type": "penalty_mode",
  "metric_params": {"penalty_mode": "heavy"},
  "expected_score": 0.4,
  "trace": { ... }
}
```

No code changes needed - tests are data-driven!
