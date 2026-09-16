# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ground truth parsing and validation utilities.

Shared logic for loading, parsing, and validating ground truth datasets
from various formats (CSV, Excel, JSON). Used by the demo backend,
notebooks, and any future CLI or API endpoints.
"""

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field

from uaef.models.ground_truth import GroundTruth
from uaef.models.tool_call import ToolCall


# Column name resolution — maps canonical field to acceptable column names
_QUESTION_ALIASES = ["question", "query", "input"]
_ANSWER_ALIASES = ["answer", "expected_answer", "expected_output", "expected"]
_CONTEXT_ALIASES = ["context"]
_TOOLS_ALIASES = ["expected_tool_calls", "tools"]


def _resolve_column(columns_lower: Dict[str, str], aliases: List[str]) -> Optional[str]:
    """Find the first matching column name from a list of aliases."""
    for alias in aliases:
        if alias in columns_lower:
            return columns_lower[alias]
    return None


class ValidationResult(BaseModel):
    """Result of ground truth validation."""

    valid: bool
    filename: str = ""
    rows: int = 0
    columns: List[str] = Field(default_factory=list)
    question_column: Optional[str] = None
    answer_column: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    preview: List[Dict[str, Any]] = Field(default_factory=list)


def parse_ground_truth_row(
    row: Dict[str, Any],
    query_only: bool = False,
) -> Union[str, Tuple[str, str, str, List[ToolCall]]]:
    """
    Extract query, expected answer, context, and expected tool calls from a
    ground truth row.

    Handles case-insensitive column names and multiple naming conventions.

    Args:
        row: Dictionary representing a single ground truth row.
        query_only: If True, only extracts and returns the query string.
                    Useful when you need the query before running the agent
                    and will build the GroundTruth separately.

    Returns:
        When query_only is False: Tuple of (query, expected_answer, context, expected_tool_calls).
        When query_only is True: query string only.
    """
    row_lower = {k.lower().strip(): v for k, v in row.items()}

    query = str(
        row_lower.get("question",
        row_lower.get("query",
        row_lower.get("input", "")))
    )

    if query_only:
        return query

    expected = str(
        row_lower.get("answer",
        row_lower.get("expected_answer",
        row_lower.get("expected_output",
        row_lower.get("expected", ""))))
    )

    raw_context = row_lower.get("context")
    context = str(raw_context) if raw_context is not None and pd.notna(raw_context) else ""

    expected_tools: List[ToolCall] = []
    raw_tools = row_lower.get("expected_tool_calls", row_lower.get("tools", None))
    # `pd.notna` returns an element-wise boolean array for list/Series inputs,
    # so guard against non-scalar values before calling it.
    is_scalar = not isinstance(raw_tools, (list, tuple, dict))
    if raw_tools and (not is_scalar or pd.notna(raw_tools)):
        try:
            parsed = json.loads(str(raw_tools)) if isinstance(raw_tools, str) else raw_tools
            if isinstance(parsed, list):
                for t in parsed:
                    expected_tools.append(ToolCall(
                        name=t.get("name", t.get("tool_name", "")),
                        arguments=t.get("arguments", t.get("parameters", {})),
                        timestamp=datetime.now(timezone.utc),
                    ))
        except (json.JSONDecodeError, TypeError):
            pass

    return query, expected, context, expected_tools


def build_ground_truth(
    row: Dict[str, Any],
    events: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, GroundTruth]:
    """
    Build a (query, GroundTruth) pair from a single ground truth row.

    If the row has a context column, that value is used. Otherwise, if
    stream events are provided, tool result contents are extracted as
    context documents across supported agent frameworks (LangGraph, Strands, etc.).

    Args:
        row: Dictionary representing a single ground truth row.
        events: Optional list of agent stream events (for ToolMessage fallback).

    Returns:
        Tuple of (query, GroundTruth).
    """
    query, expected, context, expected_tools = parse_ground_truth_row(row)

    if context:
        context_documents = [context]
    elif events:
        context_documents = _extract_tool_message_context(events)
    else:
        context_documents = []

    gt = GroundTruth(
        expected_output=expected,
        expected_tool_calls=expected_tools,
        context_documents=context_documents,
    )
    return query, gt


def build_multi_turn_ground_truth(
    turns: List[Dict[str, Any]],
    events: Optional[List[Dict[str, Any]]] = None,
) -> GroundTruth:
    """
    Build a GroundTruth from multiple conversation turns.

    Aggregates expected tool calls across all turns. The top-level
    `expected_output` is the combined per-turn expected answers (with
    turn labels), used by full-trace metrics like
    ConversationCompletenessMetric. Per-turn expected outputs are also
    stored in `expected_arguments["per_turn_expected"]` as a list so
    that per-turn evaluation can create isolated ground truths per turn.

    Context is sourced from the context column of any turn that has it;
    falls back to tool result contents from the provided events across
    supported agent frameworks (LangGraph, Strands, etc.).

    Args:
        turns: List of row dicts, one per conversation turn.
        events: Optional list of agent stream events (for ToolMessage fallback).

    Returns:
        Aggregated GroundTruth for the multi-turn session.
    """
    all_tools: List[ToolCall] = []
    context_parts: List[str] = []
    per_turn_expected: List[str] = []
    per_turn_tools: List[List[ToolCall]] = []
    expected_parts: List[str] = []

    for i, turn in enumerate(turns):
        _, expected, context, tools = parse_ground_truth_row(turn)
        all_tools.extend(tools)
        per_turn_expected.append(expected)
        per_turn_tools.append(tools)
        if expected:
            expected_parts.append(f"[Turn {i + 1}] {expected}")
        if context:
            context_parts.append(context)

    if not context_parts and events:
        context_parts = _extract_tool_message_context(events)

    combined_expected = "\n".join(expected_parts) if expected_parts else ""

    return GroundTruth(
        expected_output=combined_expected,
        expected_tool_calls=all_tools,
        context_documents=context_parts,
        expected_arguments={
            "per_turn_expected": per_turn_expected,
            "per_turn_tools": per_turn_tools,
        },
    )


def extract_context_from_trace(trace: Any) -> List[str]:
    """Extract context documents from a canonical trace's tool output.

    Framework-agnostic counterpart to :func:`_extract_tool_message_context`,
    which needs raw provider events and so needs a per-framework extractor.
    Working from the canonical trace instead means one implementation serves
    every adapter, and callers that already hold a trace (the service worker,
    for instance) don't have to keep the raw payload around to get context.

    Two sources, in order:

    1. ``ToolCall.result`` on the trace's tool calls — the intended home for
       tool output.
    2. ``Message`` entries with role ``tool`` — used by adapters that record
       tool output as conversation turns rather than on the call.

    ``MultiAgentTrace`` is walked per sub-agent, because a supervisor's own
    trace usually holds no tool calls: the work happens in the sub-agents.

    Returns an empty list when the trace carries no tool output. That is a real
    outcome, not a failure: some adapters record tool *calls* without their
    results, in which case there is nothing to ground a hallucination check on
    and the metric should report itself unavailable rather than invent context.
    """
    # MultiAgentTrace: recurse into each sub-agent, preserving agent order.
    agent_traces = getattr(trace, "agent_traces", None)
    if isinstance(agent_traces, dict):
        documents: List[str] = []
        for sub_trace in agent_traces.values():
            documents.extend(extract_context_from_trace(sub_trace))
        return documents

    documents = []
    for tool_call in getattr(trace, "tool_calls", None) or []:
        result = getattr(tool_call, "result", None)
        if result is None:
            continue
        text = result if isinstance(result, str) else str(result)
        if text.strip():
            name = getattr(tool_call, "name", "") or "tool"
            documents.append(f"[{name}] {text}")

    for message in getattr(trace, "messages", None) or []:
        role = getattr(message, "role", None)
        # MessageRole is a str enum, so compare on value to accept both forms.
        if str(getattr(role, "value", role)).lower() != "tool":
            continue
        content = getattr(message, "content", None)
        if content is None:
            continue
        text = content if isinstance(content, str) else str(content)
        if text.strip():
            documents.append(text)

    return documents


def _extract_tool_message_context(events: List[Dict[str, Any]]) -> List[str]:
    """Extract context documents from tool results in events.

    Delegates to format-specific extractors and returns the combined results.
    Supports LangGraph ToolMessage objects, Strands toolResult blocks, and
    can be extended with additional extractors for new frameworks.
    """
    extractors = [
        _extract_context_langgraph,
        _extract_context_strands,
    ]
    context_documents: List[str] = []
    for extractor in extractors:
        context_documents.extend(extractor(events))
    return context_documents


def _extract_context_langgraph(events: List[Dict[str, Any]]) -> List[str]:
    """Extract context from LangGraph ToolMessage objects in stream events."""
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        return []

    context_documents: List[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        for _key, value in event.items():
            if not isinstance(value, dict):
                continue
            msgs = value.get("messages", [])
            if not isinstance(msgs, list):
                msgs = [msgs]
            for msg in msgs:
                if isinstance(msg, ToolMessage):
                    c = msg.content
                    if isinstance(c, list):
                        context_documents.extend(str(item) for item in c)
                    else:
                        context_documents.append(str(c))
    return context_documents


def _extract_context_strands(events: List[Dict[str, Any]]) -> List[str]:
    """Extract context from Strands-style message dicts with toolResult blocks."""
    context_documents: List[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("role") != "user":
            continue
        content_blocks = event.get("content", [])
        if not isinstance(content_blocks, list):
            continue
        for block in content_blocks:
            if not isinstance(block, dict) or "toolResult" not in block:
                continue
            tr = block["toolResult"]
            if not isinstance(tr, dict):
                continue
            tr_content = tr.get("content", [])
            if isinstance(tr_content, list):
                for c in tr_content:
                    if isinstance(c, dict) and "text" in c:
                        context_documents.append(str(c["text"]))
            elif tr_content:
                context_documents.append(str(tr_content))
    return context_documents


def parse_ground_truth_dataframe(
    df: pd.DataFrame,
) -> List[Tuple[str, GroundTruth]]:
    """
    Parse an entire DataFrame into (query, GroundTruth) pairs.

    Args:
        df: DataFrame with ground truth data.

    Returns:
        List of (query_string, GroundTruth) tuples.
    """
    results = []
    for _, row in df.iterrows():
        query, gt = build_ground_truth(row.to_dict())
        results.append((query, gt))
    return results


def load_ground_truth_file(
    path: Optional[str] = None,
    content: Optional[bytes] = None,
    filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load ground truth data from a file path or raw bytes.

    Supports CSV and Excel (.xlsx, .xls) formats.

    Args:
        path: File path to load from. Mutually exclusive with content.
        content: Raw file bytes. Requires filename for format detection.
        filename: Original filename (used for format detection with content).

    Returns:
        DataFrame with ground truth data.

    Raises:
        ValueError: If neither path nor content is provided, or format is unsupported.
    """
    if path:
        p = Path(path)
        if not p.exists():
            raise ValueError(f"File not found: {path}")
        filename = p.name
        content = p.read_bytes()
    elif content is None:
        raise ValueError("Either path or content must be provided")

    if not filename:
        raise ValueError("filename is required when loading from content")

    fname_lower = filename.lower()
    if fname_lower.endswith(".xlsx") or fname_lower.endswith(".xls"):
        return pd.read_excel(io.BytesIO(content))
    elif fname_lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    elif fname_lower.endswith(".json"):
        return pd.DataFrame(json.loads(content))
    else:
        # Default to CSV
        return pd.read_csv(io.BytesIO(content))


def validate_ground_truth(
    df: pd.DataFrame,
    filename: str = "",
    preview_rows: int = 5,
) -> ValidationResult:
    """
    Validate a ground truth DataFrame and return structured results.

    Checks for required columns, detects naming conventions, and produces
    a preview of the data.

    Args:
        df: DataFrame to validate.
        filename: Original filename for reporting.
        preview_rows: Number of rows to include in the preview.

    Returns:
        ValidationResult with validation status, column mapping, warnings, and preview.
    """
    if df.empty:
        return ValidationResult(valid=False, filename=filename, errors=["File is empty"])

    columns = list(df.columns)
    row_count = len(df)

    col_lower = {c.lower().strip(): c for c in columns}
    question_col = _resolve_column(col_lower, _QUESTION_ALIASES)
    answer_col = _resolve_column(col_lower, _ANSWER_ALIASES)

    warnings = []
    if not question_col:
        warnings.append("No 'Question' column found. The first column will be used as the query.")
        question_col = columns[0]
    if not answer_col:
        warnings.append("No 'Answer' column found. Evaluation will run without ground truth comparison.")

    preview = []
    for i, row in df.head(preview_rows).iterrows():
        preview.append({
            "index": int(i) + 1,
            "question": str(row[question_col])[:200] if question_col else "",
            "answer": str(row[answer_col])[:200] if answer_col else "",
        })

    return ValidationResult(
        valid=True,
        filename=filename,
        rows=row_count,
        columns=columns,
        question_column=question_col,
        answer_column=answer_col,
        warnings=warnings,
        preview=preview,
    )
