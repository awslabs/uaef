// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from "react";

// --- Test Data Schema ---
const DATA_SCHEMA = [
  { column: "Question", required: true, aliases: ["Query", "Input"], description: "The question or prompt to send to the agent" },
  { column: "Answer", required: false, aliases: ["Expected_Answer", "Expected_Output", "Expected"], description: "The expected correct response (ground truth)" },
  { column: "context", required: false, aliases: [], description: "Reference context documents for RAG evaluation" },
  { column: "expected_tool_calls", required: false, aliases: ["tools"], description: 'Expected tool calls as JSON array, e.g. [{"name": "get_weather", "arguments": {"location": "Seattle"}}]' },
  { column: "session_id", required: false, aliases: [], description: "Groups rows into a single conversation session (required for multi-turn metrics)" },
  { column: "turn", required: false, aliases: ["turn_number", "turn#"], description: "Turn number within a session, starting from 1 (required for multi-turn metrics)" },
];

// --- Metrics Catalog ---
const METRICS = [
  // Response Quality
  { name: "answer_relevance", dimension: "Response Quality", definition: "Measures how relevant the agent's answer is to the question asked.", needsGT: false, useCases: ["rag", "conversation", "general"] },
  { name: "accuracy", dimension: "Response Quality", definition: "Measures how closely the agent's answer matches the expected output.", needsGT: true, useCases: ["rag", "reasoning", "general"] },
  { name: "completeness", dimension: "Response Quality", definition: "Measures whether the answer covers all key points from the expected output.", needsGT: true, useCases: ["rag", "reasoning", "general"] },
  { name: "coherence", dimension: "Response Quality", definition: "Measures logical consistency and flow of the response.", needsGT: false, useCases: ["conversation", "reasoning", "general"] },
  { name: "hallucination_score", dimension: "Response Quality", definition: "Detects fabricated information not supported by context or ground truth.", needsGT: true, useCases: ["rag", "safety"] },
  { name: "context_retention", dimension: "Response Quality", definition: "Measures how well the agent uses and retains provided context.", needsGT: false, useCases: ["rag", "conversation"] },

  // Tool Calling
  { name: "tool_selection_accuracy", dimension: "Tool Calling", definition: "Measures whether the agent selected the correct tool for the task.", needsGT: true, useCases: ["tool"] },
  { name: "tool_sequence_correctness", dimension: "Tool Calling", definition: "Measures whether tools were called in the correct order.", needsGT: true, useCases: ["tool"] },
  { name: "parameter_quality", dimension: "Tool Calling", definition: "Measures correctness and completeness of tool call parameters.", needsGT: true, useCases: ["tool"] },
  { name: "mcp_compliance", dimension: "Tool Calling", definition: "Checks if tool usage follows the Model Context Protocol specification.", needsGT: false, useCases: ["tool"] },

  // Safety
  { name: "safety_score", dimension: "Safety", definition: "Detects harmful, dangerous, or inappropriate content in responses.", needsGT: false, useCases: ["safety", "conversation", "general"] },
  { name: "bias_score", dimension: "Safety", definition: "Detects biased language or unfair treatment of groups.", needsGT: false, useCases: ["safety"] },
  { name: "toxicity_score", dimension: "Safety", definition: "Detects toxic, offensive, or abusive language.", needsGT: false, useCases: ["safety"] },
  { name: "prompt_injection_detection", dimension: "Safety", definition: "Measures resistance to prompt injection attacks.", needsGT: false, useCases: ["safety"] },

  // Reasoning
  { name: "chain_of_thought_coherence", dimension: "Reasoning", definition: "Measures logical coherence of step-by-step reasoning chains.", needsGT: false, useCases: ["reasoning"] },
  { name: "logical_consistency", dimension: "Reasoning", definition: "Detects contradictions within the agent's reasoning.", needsGT: false, useCases: ["reasoning"] },
  { name: "reasoning_step_correctness", dimension: "Reasoning", definition: "Validates each individual step in a reasoning chain.", needsGT: true, useCases: ["reasoning"] },
  { name: "fallacy_detection", dimension: "Reasoning", definition: "Identifies logical fallacies in the agent's arguments.", needsGT: false, useCases: ["reasoning"] },

  // Efficiency
  { name: "latency_score", dimension: "Efficiency", definition: "Scores response time relative to acceptable thresholds.", needsGT: false, useCases: ["production", "tool", "general"] },
  { name: "token_efficiency", dimension: "Efficiency", definition: "Measures how efficiently tokens are used relative to output quality.", needsGT: false, useCases: ["production"] },
  { name: "cost_efficiency", dimension: "Efficiency", definition: "Evaluates cost per request relative to quality achieved.", needsGT: false, useCases: ["production"] },

  // Conversation
  { name: "conversation_completeness", dimension: "Conversation", definition: "Measures whether the conversation achieved its intended goal.", needsGT: true, useCases: ["conversation"] },
  { name: "turn_efficiency", dimension: "Conversation", definition: "Measures whether the agent resolves tasks in minimal turns.", needsGT: false, useCases: ["conversation"] },

  // Multi-Agent
  { name: "agent_utilization", dimension: "Multi-Agent", definition: "Measures whether all agents in an orchestration are used effectively.", needsGT: false, useCases: ["multi-agent"] },
  { name: "delegation_quality", dimension: "Multi-Agent", definition: "Measures whether tasks are assigned to the most appropriate agent.", needsGT: true, useCases: ["multi-agent"] },
  { name: "workflow_completion", dimension: "Multi-Agent", definition: "Measures whether the full multi-agent workflow completes successfully.", needsGT: true, useCases: ["multi-agent"] },
  { name: "coordination_efficiency", dimension: "Multi-Agent", definition: "Measures how well agents coordinate and avoid redundant work.", needsGT: false, useCases: ["multi-agent"] },

  // RAGAS
  { name: "ragas_faithfulness", dimension: "RAGAS", definition: "Measures if all claims in the answer are supported by the provided context.", needsGT: true, useCases: ["rag"] },
  { name: "ragas_context_precision", dimension: "RAGAS", definition: "Measures if the most relevant context chunks are ranked highest.", needsGT: true, useCases: ["rag"] },
  { name: "ragas_context_recall", dimension: "RAGAS", definition: "Measures if the retrieved context covers all parts of the ground truth.", needsGT: true, useCases: ["rag"] },
  { name: "ragas_answer_correctness", dimension: "RAGAS", definition: "Measures factual correctness of the answer against ground truth.", needsGT: true, useCases: ["rag"] },
  { name: "ragas_tool_call_accuracy", dimension: "RAGAS", definition: "Measures overall correctness of tool calls (selection + parameters).", needsGT: true, useCases: ["tool"] },

  // DeepEval
  { name: "deepeval_contextual_relevancy", dimension: "DeepEval", definition: "Measures if the retrieved context is relevant to the query.", needsGT: false, useCases: ["rag"] },
  { name: "deepeval_hallucination", dimension: "DeepEval", definition: "Detects claims in the response not supported by the context.", needsGT: true, useCases: ["rag", "safety"] },
  { name: "deepeval_faithfulness", dimension: "DeepEval", definition: "Measures if the response is factually consistent with the context.", needsGT: true, useCases: ["rag"] },
  { name: "deepeval_answer_relevancy", dimension: "DeepEval", definition: "Measures if the answer addresses the original question.", needsGT: false, useCases: ["rag", "general"] },
  { name: "deepeval_tool_correctness", dimension: "DeepEval", definition: "Measures tool selection and usage correctness.", needsGT: true, useCases: ["tool"] },
];

const USE_CASES = [
  { id: "all", label: "All" },
  { id: "rag", label: "RAG / Knowledge Q&A" },
  { id: "tool", label: "Tool-Using Agent" },
  { id: "conversation", label: "Multi-Turn Conversation" },
  { id: "reasoning", label: "Reasoning / CoT" },
  { id: "safety", label: "Safety & Compliance" },
  { id: "multi-agent", label: "Multi-Agent" },
  { id: "production", label: "Production Monitoring" },
  { id: "general", label: "General Q&A" },
];

export default function TaxonomyPanel() {
  const [useCaseFilter, setUseCaseFilter] = useState("all");
  const [gtFilter, setGtFilter] = useState("all"); // "all", "yes", "no"
  const [searchTerm, setSearchTerm] = useState("");

  const filtered = METRICS.filter((m) => {
    if (useCaseFilter !== "all" && !m.useCases.includes(useCaseFilter)) return false;
    if (gtFilter === "yes" && !m.needsGT) return false;
    if (gtFilter === "no" && m.needsGT) return false;
    if (searchTerm && !m.name.includes(searchTerm.toLowerCase()) && !m.definition.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  // Group by dimension for display
  const grouped = {};
  for (const m of filtered) {
    if (!grouped[m.dimension]) grouped[m.dimension] = [];
    grouped[m.dimension].push(m);
  }

  return (
    <>
      {/* Test Data Schema */}
      <div className="card">
        <h2>Test Data Schema</h2>
        <p className="hint">
          Your ground truth file should be a CSV or Excel file with these columns.
          Column names are case-insensitive. Any listed alias is accepted interchangeably.
        </p>
        <table>
          <thead>
            <tr>
              <th>Required</th>
              <th>Accepted Names</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            {DATA_SCHEMA.map((col) => (
              <tr key={col.column}>
                <td>
                  <span className={`tag ${col.required ? "tag-blue" : "tag-yellow"}`}>
                    {col.required ? "Required" : "Optional"}
                  </span>
                </td>
                <td>
                  <code style={{ marginRight: 4, fontWeight: 600 }}>{col.column}</code>
                  {col.aliases.map((a) => <code key={a} style={{ marginRight: 4 }}>{a}</code>)}
                </td>
                <td>{col.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Metrics Catalog */}
      <div className="card">
        <h2>Metrics Catalog</h2>
        <p className="hint">
          Filter metrics by use case and ground truth requirement to find the right ones for your evaluation.
        </p>

        <div className="form-row" style={{ marginTop: 12, marginBottom: 16, flexWrap: "wrap", gap: 12 }}>
          <div className="form-group" style={{ flex: "1 1 200px", minWidth: 160 }}>
            <label>Use Case</label>
            <select value={useCaseFilter} onChange={(e) => setUseCaseFilter(e.target.value)}>
              {USE_CASES.map((uc) => (
                <option key={uc.id} value={uc.id}>{uc.label}</option>
              ))}
            </select>
          </div>
          <div className="form-group" style={{ flex: "1 1 200px", minWidth: 160 }}>
            <label>Ground Truth Needed</label>
            <select value={gtFilter} onChange={(e) => setGtFilter(e.target.value)}>
              <option value="all">All</option>
              <option value="yes">Yes — requires GT</option>
              <option value="no">No — works without GT</option>
            </select>
          </div>
          <div className="form-group" style={{ flex: "2 1 250px", minWidth: 200 }}>
            <label>Search</label>
            <input
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Filter by name or description..."
            />
          </div>
        </div>

        <p className="hint" style={{ marginBottom: 8 }}>
          Showing {filtered.length} of {METRICS.length} metrics
        </p>
      </div>

      {Object.entries(grouped).map(([dimension, metrics]) => (
        <div className="card" key={dimension} style={{ paddingTop: 12 }}>
          <h3 style={{ marginBottom: 8 }}>{dimension}</h3>
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Definition</th>
                <th>GT Needed</th>
                <th>Use Cases</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m) => (
                <tr key={m.name}>
                  <td><code className="taxonomy-metric-name">{m.name}</code></td>
                  <td style={{ fontSize: "0.85rem" }}>{m.definition}</td>
                  <td>
                    <span className={`tag ${m.needsGT ? "tag-orange" : "tag-blue"}`}>
                      {m.needsGT ? "Yes" : "No"}
                    </span>
                  </td>
                  <td style={{ fontSize: "0.75rem" }}>
                    {m.useCases.map((uc) => (
                      <span key={uc} className="tag" style={{ marginRight: 3, marginBottom: 2 }}>{uc}</span>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}

      {filtered.length === 0 && (
        <div className="card">
          <p className="empty">No metrics match your filters.</p>
        </div>
      )}
    </>
  );
}
