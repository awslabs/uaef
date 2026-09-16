// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState, useEffect } from "react";
import { api } from "../api";

function scoreColor(v) {
  if (v >= 0.8) return "var(--green)";
  if (v >= 0.6) return "var(--orange)";
  return "var(--red)";
}

const COLUMNS = [
  { key: "experiment_id", label: "Experiment ID" },
  { key: "experiment_name", label: "Name" },
  { key: "experiment_objective", label: "Objective" },
  { key: "agent_name", label: "Agent" },
  { key: "dataset_name", label: "Dataset" },
  { key: "created_at", label: "Created" },
  { key: "result_path", label: "Results", sortable: false },
];

export default function ExperimentsPanel({ onViewResults }) {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [sortKey, setSortKey] = useState("created_at");
  const [sortDir, setSortDir] = useState("desc");

  const load = () => {
    setLoading(true);
    setError("");
    api.getExperiments()
      .then((data) => setExperiments(data.experiments || []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const sorted = [...experiments].sort((a, b) => {
    let aVal = a[sortKey] || "";
    let bVal = b[sortKey] || "";
    if (typeof aVal === "string") aVal = aVal.toLowerCase();
    if (typeof bVal === "string") bVal = bVal.toLowerCase();
    if (aVal < bVal) return sortDir === "asc" ? -1 : 1;
    if (aVal > bVal) return sortDir === "asc" ? 1 : -1;
    return 0;
  });

  if (loading) return <p className="loading">Loading experiments...</p>;

  if (error) {
    return (
      <div className="card">
        <p className="hint">Could not load experiments: {error}</p>
        <p className="hint">
          Experiments appear here after you run an evaluation with the persist
          toggle enabled. Make sure AWS credentials are configured.
        </p>
      </div>
    );
  }

  if (experiments.length === 0) {
    return (
      <div className="card">
        <p className="empty">
          No experiments yet. Run an evaluation with the persist toggle enabled
          to save results here.
        </p>
      </div>
    );
  }

  return (
    <>
      <div className="card">
        <div className="card-header-row">
          <h2>Experiments ({experiments.length})</h2>
          <button className="btn btn-outline btn-sm" onClick={load}>Refresh</button>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                {COLUMNS.map((col) => (
                  <th
                    key={col.key}
                    onClick={col.sortable !== false ? () => handleSort(col.key) : undefined}
                    style={col.sortable !== false ? { cursor: "pointer", userSelect: "none" } : {}}
                  >
                    {col.label}
                    {col.sortable !== false && sortKey === col.key && (
                      <span className="sort-arrow">{sortDir === "asc" ? " ▲" : " ▼"}</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((exp) => (
                <tr key={exp.experiment_id}>
                  <td className="cell-copy" onClick={() => navigator.clipboard.writeText(exp.experiment_id)} title={exp.experiment_id}>
                    <code style={{ fontSize: "0.7rem" }}>{exp.experiment_id.slice(0, 12)}...</code>
                  </td>
                  <td className="cell-copy" style={{ fontWeight: 600 }} onClick={() => navigator.clipboard.writeText(exp.experiment_name)} title={exp.experiment_name}>
                    {exp.experiment_name}
                  </td>
                  <td className="cell-copy cell-expandable" onClick={() => navigator.clipboard.writeText(exp.experiment_objective || "")} title={exp.experiment_objective || ""}>
                    {exp.experiment_objective || "—"}
                  </td>
                  <td>
                    <span className="tag tag-blue">{exp.agent_name || exp.framework}</span>
                  </td>
                  <td className="cell-copy" onClick={() => navigator.clipboard.writeText(exp.dataset_s3_path || exp.dataset_name || "")} title={exp.dataset_s3_path || ""}>
                    {exp.dataset_s3_path ? (
                      <a
                        href={exp.dataset_s3_path.replace(
                          /^s3:\/\/([^/]+)\/(.+)$/,
                          "https://s3.console.aws.amazon.com/s3/object/$1?prefix=$2"
                        )}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="dataset-link"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {exp.dataset_name || "dataset"}
                      </a>
                    ) : (
                      <span>{exp.dataset_name || "—"}</span>
                    )}
                  </td>
                  <td className="cell-copy" style={{ fontSize: "0.75rem", color: "var(--muted)" }} onClick={() => navigator.clipboard.writeText(exp.created_at || "")} title={exp.created_at}>
                    {exp.created_at ? new Date(exp.created_at).toLocaleString() : "—"}
                  </td>
                  <td>
                    {exp.result_path ? (
                      <>
                        <button
                          className="btn btn-primary btn-sm"
                          style={{ marginRight: 6 }}
                          onClick={() => onViewResults && onViewResults(exp.experiment_id)}
                        >
                          View
                        </button>
                        <a
                          href={exp.result_path.replace(
                            /^s3:\/\/([^/]+)\/(.+)$/,
                            "https://s3.console.aws.amazon.com/s3/object/$1?prefix=$2"
                          )}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="btn btn-outline btn-sm"
                        >
                          S3 ↗
                        </a>
                      </>
                    ) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
