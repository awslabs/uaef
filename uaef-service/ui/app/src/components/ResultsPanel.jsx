// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";
import { openExperimentReport } from "../report";

function scoreColor(v) {
  if (v >= 0.8) return "var(--green)";
  if (v >= 0.6) return "var(--orange)";
  return "var(--red)";
}

export default function ResultsPanel({ result }) {
  const [expandedRow, setExpandedRow] = useState(null);

  if (!result) {
    return (
      <p className="empty">
        Run an evaluation in the Evaluate tab to see results here.
      </p>
    );
  }

  const {
    experiment_name, experiment_objective, experiment_id,
    framework, evaluation_count, total_queries,
    successful_queries, failed_queries,
    overall_average_score, average_scores, rows, errors,
    dimension_metrics,
  } = result;

  const scoreEntries = Object.entries(average_scores || {}).sort(([, a], [, b]) => b - a);

  // Radar axes come from the backend's real dimension -> metric grouping.
  // Falls back to keyword matching only for results stored before that field
  // existed; that heuristic mis-filed metrics whose names lack the keyword
  // (e.g. role_adherence -> "Reasoning", agent_tone -> "Multi-Agent").
  const dimMap = {};
  if (dimension_metrics && Object.keys(dimension_metrics).length > 0) {
    for (const [dim, names] of Object.entries(dimension_metrics)) {
      const scores = (names || [])
        .map((n) => average_scores?.[n])
        .filter((v) => typeof v === "number");
      if (scores.length > 0) dimMap[dim] = scores;
    }
  } else {
    for (const [k, v] of scoreEntries) {
      const dim =
        k.includes("tool") || k.includes("mcp") || k.includes("parameter") ? "Tool Calling"
        : k.includes("accuracy") || k.includes("relevance") || k.includes("completeness") || k.includes("hallucination") ? "Response Quality"
        : k.includes("safety") || k.includes("bias") || k.includes("toxicity") || k.includes("injection") ? "Responsible AI"
        : k.includes("latency") || k.includes("token") || k.includes("cost") || k.includes("throughput") ? "Performance"
        : k.includes("context") || k.includes("coherence") || k.includes("conversation") || k.includes("turn") ? "Multi-Turn"
        : k.includes("agent") || k.includes("delegation") || k.includes("workflow") || k.includes("coordination") ? "Multi-Agent"
        : "Reasoning";
      if (!dimMap[dim]) dimMap[dim] = [];
      dimMap[dim].push(v);
    }
  }
  const radarData = Object.entries(dimMap).map(([dim, scores]) => ({
    dimension: dim,
    score: +(scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(3),
  }));

  return (
    <>
      {/* Summary */}
      <div className="card">
        <div className="card-header-row">
          <h2>{experiment_name || "Evaluation Results"}</h2>
          {experiment_id && (
            <button className="btn btn-outline btn-sm" onClick={() => openExperimentReport(experiment_id)}>
              📊 Open full report
            </button>
          )}
        </div>
        {experiment_objective && <p className="hint">{experiment_objective}</p>}
        <div className="summary-row">
          <div className="summary-item">
            <div className="summary-value" style={{ color: scoreColor(overall_average_score) }}>
              {(overall_average_score * 100).toFixed(1)}%
            </div>
            <div className="summary-label">Overall Score</div>
          </div>
          <div className="summary-item">
            <div className="summary-value">{evaluation_count}</div>
            <div className="summary-label">Evaluated</div>
          </div>
          <div className="summary-item">
            <div className="summary-value">{scoreEntries.length}</div>
            <div className="summary-label">Metrics</div>
          </div>
          <div className="summary-item">
            <div className="summary-value">
              <span className="tag tag-blue">{framework}</span>
            </div>
            <div className="summary-label">Framework</div>
          </div>
          {failed_queries > 0 && (
            <div className="summary-item">
              <div className="summary-value" style={{ color: "var(--red)" }}>{failed_queries}</div>
              <div className="summary-label">Failed</div>
            </div>
          )}
          {experiment_id && (
            <div className="summary-item">
              <div className="summary-value" style={{ fontSize: "0.75rem" }}>{experiment_id.slice(0, 8)}...</div>
              <div className="summary-label">Experiment ID</div>
            </div>
          )}
        </div>
      </div>

      {/* Metric averages */}
      <div className="card">
        <h2>Average Metric Scores</h2>
        <div className="metric-grid">
          {scoreEntries.map(([k, v]) => (
            <div className="metric-card" key={k}>
              <div className="value" style={{ color: scoreColor(v) }}>{(v * 100).toFixed(1)}%</div>
              <div className="label">{k}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Charts */}
      <div className="charts-grid">
        <div className="card chart-card">
          <h3>{radarData.length >= 3 ? "Dimension Radar" : "Dimension Scores"}</h3>
          {/* A radar needs three axes to enclose an area; with one or two
              dimensions recharts draws a degenerate shape that looks like a
              blank card. Show the numbers directly instead. */}
          {radarData.length >= 3 ? (
            <ResponsiveContainer width="100%" height={380}>
              <RadarChart data={radarData} outerRadius="80%">
                <PolarGrid />
                <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 12 }} />
                <PolarRadiusAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                <Radar dataKey="score" stroke="#4f46e5" fill="#4f46e5" fillOpacity={0.3} />
              </RadarChart>
            </ResponsiveContainer>
          ) : radarData.length > 0 ? (
            <>
              <p className="hint">
                {radarData.length === 1 ? "One dimension" : "Two dimensions"} evaluated.
                A radar chart needs at least three.
              </p>
              <div className="metric-grid">
                {radarData.map((d) => (
                  <div className="metric-card" key={d.dimension}>
                    <div className="value" style={{ color: scoreColor(d.score) }}>
                      {(d.score * 100).toFixed(1)}%
                    </div>
                    <div className="label">{d.dimension}</div>
                  </div>
                ))}
              </div>
            </>
          ) : <p className="empty">No data</p>}
        </div>
        <div className="card chart-card">
          <h3>Metric Scores</h3>
          {scoreEntries.length > 0 ? (
            <ResponsiveContainer width="100%" height={Math.max(250, scoreEntries.length * 26)}>
              <BarChart data={scoreEntries.map(([k, v]) => ({ name: k, score: +v.toFixed(3) }))}
                layout="vertical" margin={{ left: 140 }}>
                <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={140} />
                <Tooltip />
                <Bar dataKey="score">
                  {scoreEntries.map(([, v], i) => (
                    <Cell key={i} fill={v >= 0.8 ? "#16a34a" : v >= 0.6 ? "#ea580c" : "#dc2626"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <p className="empty">No data</p>}
        </div>
      </div>

      {/* Per-test-case table */}
      {rows && rows.length > 0 && (
        <div className="card">
          <h3>Per-Question Evaluation Results</h3>
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Query</th>
                  <th>Agent Response</th>
                  <th>Expected</th>
                  <th>Score</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <React.Fragment key={i}>
                    <tr>
                      <td>{r.index}</td>
                      <td className="cell-truncate">{r.query}</td>
                      <td className="cell-truncate">{r.agent_response}</td>
                      <td className="cell-truncate">{r.expected_answer}</td>
                      <td style={{ color: scoreColor(r.overall_score), fontWeight: 600 }}>
                        {(r.overall_score * 100).toFixed(1)}%
                      </td>
                      <td>
                        <span className={`tag ${r.passed ? "tag-green" : "tag-red"}`}>
                          {r.passed ? "PASS" : "FAIL"}
                        </span>
                      </td>
                      <td>
                        <button
                          className="btn btn-outline btn-sm"
                          onClick={() => setExpandedRow(expandedRow === i ? null : i)}
                        >
                          {expandedRow === i ? "▲" : "▼"}
                        </button>
                      </td>
                    </tr>
                    {expandedRow === i && (
                      <tr>
                        <td colSpan={7} style={{ background: "var(--bg)", padding: 16 }}>
                          <div className="expanded-row">
                            {/* Multi-turn rows cover a whole conversation, so show
                                each turn rather than only the final response. */}
                            {r.turns && r.turns.length > 0 ? (
                              <div className="expanded-section">
                                <strong>Conversation ({r.turns.length} turns):</strong>
                                <div className="turn-list">
                                  {r.turns.map((t) => (
                                    <div key={t.turn} className="turn-item">
                                      <div className="turn-head">
                                        <span className="turn-badge">Turn {t.turn}</span>
                                        {Object.entries(t.metric_scores || {}).map(([k, v]) => (
                                          <span
                                            key={k}
                                            className="mini-score"
                                            style={{ borderColor: scoreColor(v) }}
                                          >
                                            {k}: {(v * 100).toFixed(0)}%
                                          </span>
                                        ))}
                                      </div>
                                      <div className="turn-line">
                                        <span className="turn-label">User</span>
                                        <p>{t.query}</p>
                                      </div>
                                      <div className="turn-line">
                                        <span className="turn-label">Agent</span>
                                        <p>{t.response || <em>no response</em>}</p>
                                      </div>
                                      {t.expected && (
                                        <div className="turn-line">
                                          <span className="turn-label">Expected</span>
                                          <p>{t.expected}</p>
                                        </div>
                                      )}
                                      {t.tool_calls && t.tool_calls.length > 0 && (
                                        <div className="turn-line">
                                          <span className="turn-label">Tools</span>
                                          <p>{t.tool_calls.join(", ")}</p>
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ) : (
                              <>
                                <div className="expanded-section">
                                  <strong>Agent Response (full):</strong>
                                  <p>{r.agent_response}</p>
                                </div>
                                <div className="expanded-section">
                                  <strong>Expected Answer:</strong>
                                  <p>{r.expected_answer}</p>
                                </div>
                              </>
                            )}
                            <div className="expanded-section">
                              <strong>Dimension Scores:</strong>
                              <div className="mini-scores">
                                {Object.entries(r.dimension_scores || {}).map(([k, v]) => (
                                  <span key={k} className="mini-score" style={{ borderColor: scoreColor(v) }}>
                                    {k}: {(v * 100).toFixed(0)}%
                                  </span>
                                ))}
                              </div>
                            </div>
                            <div className="expanded-section">
                              <strong>Metric Scores:</strong>
                              <div className="mini-scores">
                                {Object.entries(r.metric_scores || {}).map(([k, v]) => (
                                  <span key={k} className="mini-score" style={{ borderColor: scoreColor(v) }}>
                                    {k}: {(v * 100).toFixed(0)}%
                                  </span>
                                ))}
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Errors */}
      {errors && errors.length > 0 && (
        <div className="card">
          <h3 style={{ color: "var(--red)" }}>Failed Queries ({errors.length})</h3>
          <table>
            <thead>
              <tr><th>#</th><th>Query</th><th>Error</th></tr>
            </thead>
            <tbody>
              {errors.map((e, i) => (
                <tr key={i}>
                  <td>{e.index}</td>
                  <td className="cell-truncate">{e.query}</td>
                  <td style={{ color: "var(--red)", fontSize: "0.8rem" }}>{e.error}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
