// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState, useEffect, useMemo } from "react";
import { api } from "../api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";

const COLORS = ["#4f46e5", "#16a34a", "#ea580c", "#0891b2", "#7c3aed", "#dc2626", "#ca8a04"];

function scoreColor(v) {
  if (v >= 0.8) return "var(--green)";
  if (v >= 0.6) return "var(--orange)";
  return "var(--red)";
}

export default function ComparePanel() {
  const [experiments, setExperiments] = useState([]);
  const [selected, setSelected] = useState(new Set());
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Filters
  const [search, setSearch] = useState("");
  const [filterFramework, setFilterFramework] = useState("");
  const [filterDataset, setFilterDataset] = useState("");

  useEffect(() => {
    api.getExperiments().then((d) => setExperiments(d.experiments || [])).catch(() => {});
  }, []);

  // Derive unique values for filter dropdowns
  const frameworks = useMemo(() => [...new Set(experiments.map((e) => e.agent_name || e.framework).filter(Boolean))].sort(), [experiments]);
  const datasets = useMemo(() => [...new Set(experiments.map((e) => e.dataset_name).filter((d) => d && d !== "\u2014"))].sort(), [experiments]);

  // Filtered experiments
  const filtered = useMemo(() => {
    return experiments.filter((e) => {
      if (search && !e.experiment_name.toLowerCase().includes(search.toLowerCase()) && !e.experiment_id.includes(search)) return false;
      if (filterFramework && (e.agent_name || e.framework) !== filterFramework) return false;
      if (filterDataset && e.dataset_name !== filterDataset) return false;
      return true;
    });
  }, [experiments, search, filterFramework, filterDataset]);

  const toggle = (id) => {
    setSelected((p) => { const n = new Set(p); n.has(id) ? n.delete(id) : n.add(id); return n; });
    setResult(null);
  };

  const run = async () => {
    setError(""); setLoading(true); setResult(null);
    try { setResult(await api.compareExperiments([...selected])); }
    catch (e) { setError(e.message); }
    setLoading(false);
  };

  const chart = result ? (result.table || []).filter((r) => r.spread > 0).sort((a, b) => b.spread - a.spread).slice(0, 15) : [];

  return (
    <div>
      <div className="card">
        <h2>Compare Experiments</h2>

        <div className="compare-filters">
          <input
            className="compare-search"
            placeholder="Search by name or ID..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select value={filterFramework} onChange={(e) => setFilterFramework(e.target.value)}>
            <option value="">All Agents</option>
            {frameworks.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
          <select value={filterDataset} onChange={(e) => setFilterDataset(e.target.value)}>
            <option value="">All Datasets</option>
            {datasets.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
          <span className="hint">{filtered.length} of {experiments.length} experiments</span>
        </div>

        {selected.size > 0 && (
          <div className="compare-selected-bar">
            <span>{selected.size} selected</span>
            <button className="btn btn-outline btn-sm" onClick={() => { setSelected(new Set()); setResult(null); }}>Clear</button>
          </div>
        )}

        <div className="compare-select-list">
          {filtered.slice(0, 50).map((e) => (
            <label key={e.experiment_id} className={"compare-select-item" + (selected.has(e.experiment_id) ? " compare-item-selected" : "")}>
              <input type="checkbox" checked={selected.has(e.experiment_id)} onChange={() => toggle(e.experiment_id)} />
              <span className="compare-select-name">{e.experiment_name}</span>
              <span className="tag tag-blue">{e.agent_name || e.framework}</span>
              {e.dataset_name && e.dataset_name !== "\u2014" && (
                <span className="compare-select-dataset">{e.dataset_name}</span>
              )}
              <span style={{ color: scoreColor(e.overall_average_score), fontWeight: 600, fontSize: "0.8rem" }}>
                {(e.overall_average_score * 100).toFixed(1)}%
              </span>
            </label>
          ))}
          {filtered.length > 50 && (
            <p className="hint" style={{ padding: 8 }}>Showing first 50 of {filtered.length}. Use filters to narrow down.</p>
          )}
          {filtered.length === 0 && (
            <p className="hint" style={{ padding: 12 }}>No experiments match your filters.</p>
          )}
        </div>

        <div style={{ marginTop: 12, display: "flex", gap: 8, alignItems: "center" }}>
          <button className="btn btn-primary" onClick={run} disabled={loading || selected.size < 2}>
            {loading ? "Comparing..." : "Compare " + selected.size + " Experiments"}
          </button>
          {selected.size < 2 && <span className="hint">Select at least 2</span>}
          {error && <span className="error">{error}</span>}
        </div>
      </div>

      {result && (
        <div>
          <div className="card">
            <h2>Overall Scores</h2>
            <div className="compare-scores-row">
              {result.experiments.map((exp, i) => (
                <div key={exp.id} className="compare-score-card" style={{ borderTopColor: COLORS[i % COLORS.length] }}>
                  <div style={{ fontSize: "1.5rem", fontWeight: 700, color: scoreColor(exp.overall_score) }}>
                    {(exp.overall_score * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: "0.8rem", fontWeight: 600 }}>{exp.name}</div>
                  <span className="tag tag-blue">{exp.agent_name || exp.framework}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <h2>Insights</h2>
            <div className="insights-list">
              {result.insights.map((ins, i) => <div key={i} className="insight-item">{ins}</div>)}
            </div>
          </div>

          {chart.length > 0 && (
            <div className="card chart-card">
              <h3>Most Variable Metrics</h3>
              <ResponsiveContainer width="100%" height={Math.max(300, chart.length * 35)}>
                <BarChart data={chart} layout="vertical" margin={{ left: 160 }}>
                  <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10 }} />
                  <YAxis type="category" dataKey="metric" tick={{ fontSize: 10 }} width={160} />
                  <Tooltip formatter={(v) => v != null ? (v * 100).toFixed(1) + "%" : "N/A"} />
                  <Legend />
                  {result.experiments.map((exp, i) => (
                    <Bar key={exp.id} dataKey={"exp_" + i} name={exp.name} fill={COLORS[i % COLORS.length]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="card">
            <h3>Metric-by-Metric Comparison</h3>
            <div style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    {result.experiments.map((exp, i) => (
                      <th key={exp.id} style={{ color: COLORS[i % COLORS.length] }}>{exp.name}</th>
                    ))}
                    <th>Spread</th>
                  </tr>
                </thead>
                <tbody>
                  {result.table.map((row) => (
                    <tr key={row.metric}>
                      <td>{row.metric}</td>
                      {result.experiments.map((_, i) => {
                        const val = row["exp_" + i];
                        const best = val === row.max && row.spread > 0.05;
                        const worst = val === row.min && row.spread > 0.05;
                        return (
                          <td key={i} style={{ fontWeight: best || worst ? 600 : 400, color: best ? "var(--green)" : worst ? "var(--red)" : "inherit" }}>
                            {val != null ? (val * 100).toFixed(1) + "%" : "\u2014"}
                          </td>
                        );
                      })}
                      <td style={{ color: row.spread > 0.1 ? "var(--red)" : row.spread > 0.05 ? "var(--orange)" : "var(--muted)", fontWeight: 600 }}>
                        {(row.spread * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
