// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState, useEffect } from "react";
import { api } from "../api";

// Metrics that require ground truth (Answer column) to work
const GT_REQUIRED_METRICS = new Set([
  "accuracy", "completeness", "hallucination_score",
  "tool_selection_accuracy", "tool_sequence_correctness", "parameter_quality",
  "reasoning_step_correctness", "conversation_completeness",
  "delegation_quality", "workflow_completion",
  "ragas_faithfulness", "ragas_context_precision", "ragas_context_recall",
  "ragas_answer_precision", "ragas_answer_recall", "ragas_answer_correctness",
  "ragas_tool_call_accuracy",
  "deepeval_hallucination", "deepeval_faithfulness", "deepeval_tool_correctness",
  "stickler_overall_score", "stickler_field_precision", "stickler_field_recall",
  "stickler_field_f1", "stickler_false_alarm_rate", "stickler_false_discovery_rate",
]);

export default function SetupPanel({ metrics, onComplete }) {
  const [agentTypes, setAgentTypes] = useState([]);
  const [framework, setFramework] = useState("agentcore");
  const [file, setFile] = useState(null);
  const [dataSource, setDataSource] = useState("upload"); // "upload" or "s3"
  const [s3DataPath, setS3DataPath] = useState("");

  // Dynamic agent connection fields (populated from agentTypes config)
  const [connectionFields, setConnectionFields] = useState({});

  // AgentCore runtimes
  const [agentCoreRuntimes, setAgentCoreRuntimes] = useState([]);
  const [loadingRuntimes, setLoadingRuntimes] = useState(false);

  // Metrics
  const [selectedMetrics, setSelectedMetrics] = useState(null);

  // Experiment
  const [persist, setPersist] = useState(false);
  const [experimentName, setExperimentName] = useState("");
  const [experimentObjective, setExperimentObjective] = useState("");

  // State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [expandedDims, setExpandedDims] = useState(new Set());
  const [validation, setValidation] = useState(null);
  const [validating, setValidating] = useState(false);
  const [metricGtFilter, setMetricGtFilter] = useState("all");

  useEffect(() => {
    api.getAgentTypes().then(setAgentTypes).catch(() => {});
  }, []);

  // Reset connection fields when framework changes
  useEffect(() => {
    const currentType = agentTypes.find((t) => t.id === framework);
    if (currentType) {
      const defaults = {};
      for (const f of currentType.fields) {
        defaults[f.name] = f.default || "";
      }
      setConnectionFields(defaults);
    }
  }, [framework, agentTypes]);

  // Fetch AgentCore runtimes when agentcore is selected
  useEffect(() => {
    if (framework === "agentcore") {
      setLoadingRuntimes(true);
      const region = connectionFields.region || "us-east-1";
      api.getAgentCoreRuntimes(region)
        .then((data) => setAgentCoreRuntimes(data.runtimes || []))
        .catch(() => setAgentCoreRuntimes([]))
        .finally(() => setLoadingRuntimes(false));

      // Also load .env.agentcore values if available
      api.getAgentCoreEnv().then((env) => {
        if (env.found) {
          setConnectionFields((prev) => ({
            ...prev,
            agent_runtime_arn: prev.agent_runtime_arn || env.agent_runtime_arn || "",
            bearer_token: prev.bearer_token || env.bearer_token || "",
          }));
        }
      }).catch(() => {});
    }
  }, [framework]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (selectedMetrics === null && Object.keys(metrics).length > 0) {
      setSelectedMetrics(new Set(Object.values(metrics).flat()));
    }
  }, [metrics, selectedMetrics]);

  const allMetrics = Object.values(metrics).flat();
  const currentType = agentTypes.find((t) => t.id === framework);

  const toggleMetric = (m) =>
    setSelectedMetrics((prev) => {
      const next = new Set(prev);
      next.has(m) ? next.delete(m) : next.add(m);
      return next;
    });

  const toggleDimension = (dim) => {
    const mets = metrics[dim] || [];
    setSelectedMetrics((prev) => {
      const next = new Set(prev);
      const allSelected = mets.every((m) => next.has(m));
      mets.forEach((m) => (allSelected ? next.delete(m) : next.add(m)));
      return next;
    });
  };

  const handleEvaluate = async () => {
    if (!file && dataSource === "upload") { setError("Upload a ground truth file"); return; }
    if (!s3DataPath && dataSource === "s3") { setError("Enter an S3 path"); return; }

    // Validate required connection fields
    const requiredFields = (currentType?.fields || []).filter((f) => !f.optional);
    for (const f of requiredFields) {
      if (!connectionFields[f.name]) {
        setError(`${f.label} is required`);
        return;
      }
    }

    setError("");
    setLoading(true);
    setStatus("Sending queries to agent and evaluating...");

    try {
      const fd = new FormData();
      if (dataSource === "upload" && file) {
        fd.append("file", file);
        fd.append("data_source", "upload");
      } else {
        fd.append("s3_data_path", s3DataPath);
        fd.append("data_source", "s3");
      }
      fd.append("framework", framework);
      fd.append("metrics", selectedMetrics ? [...selectedMetrics].join(",") : "");
      fd.append("persist", persist ? "true" : "false");

      // Append all connection fields
      for (const [key, value] of Object.entries(connectionFields)) {
        if (value) fd.append(key, value);
      }

      if (persist && experimentName) fd.append("experiment_name", experimentName);
      if (persist && experimentObjective) fd.append("experiment_objective", experimentObjective);

      // Derive agent name from the selected agent type/runtime
      let derivedAgentName = "";
      if (framework === "agentcore" && connectionFields.agent_runtime_arn) {
        const runtime = agentCoreRuntimes.find((r) => r.arn === connectionFields.agent_runtime_arn);
        derivedAgentName = runtime ? runtime.name : connectionFields.agent_runtime_arn;
      } else {
        const selectedType = agentTypes.find((t) => t.id === framework);
        derivedAgentName = selectedType ? selectedType.label : framework;
      }
      if (derivedAgentName) fd.append("agent_name", derivedAgentName);

      const result = await api.evaluate(fd);
      setStatus("");
      onComplete(result);
    } catch (e) {
      setError(e.message);
      setStatus("");
    }
    setLoading(false);
  };

  return (
    <>
      {/* Step 1: Evaluation data */}
      <div className="card">
        <h2>1. Upload Evaluation Data</h2>
        <p className="hint">
          Provide your test data (questions + expected answers). Upload a file or point to an existing S3 path.
          Expected columns: <code>Question</code>, <code>Answer</code>.
        </p>
        <div className="form-row" style={{ alignItems: "end" }}>
          <div className="form-group" style={{ flex: "0 0 auto", minWidth: 160 }}>
            <label>Source</label>
            <select value={dataSource} onChange={(e) => { setDataSource(e.target.value); setValidation(null); }}>
              <option value="upload">Upload File</option>
              <option value="s3">S3 Path</option>
            </select>
          </div>
          {dataSource === "upload" ? (
            <div className="form-group" style={{ flex: 2 }}>
              <label>Ground Truth File</label>
              <input
                type="file"
                accept=".xlsx,.xls,.csv"
                onChange={(e) => { setFile(e.target.files[0]); setValidation(null); }}
              />
            </div>
          ) : (
            <div className="form-group" style={{ flex: 2 }}>
              <label>S3 Path</label>
              <input
                value={s3DataPath}
                onChange={(e) => { setS3DataPath(e.target.value); setValidation(null); }}
                placeholder="s3://my-bucket/data/ground-truth.xlsx"
              />
            </div>
          )}
          <button
            className="btn btn-outline"
            style={{ alignSelf: "flex-end" }}
            disabled={validating || (dataSource === "upload" ? !file : !s3DataPath)}
            onClick={async () => {
              setValidating(true);
              setValidation(null);
              try {
                const fd = new FormData();
                if (dataSource === "upload") {
                  fd.append("file", file);
                  fd.append("source", "upload");
                } else {
                  fd.append("s3_path", s3DataPath);
                  fd.append("source", "s3");
                }
                const res = await api.validateData(fd);
                setValidation(res);
              } catch (e) {
                setValidation({ valid: false, error: e.message });
              }
              setValidating(false);
            }}
          >
            {validating ? "Validating..." : "Validate"}
          </button>
        </div>

        {validation && (
          <div className={`validation-result ${validation.valid ? "validation-ok" : "validation-err"}`}>
            {validation.valid ? (
              <>
                <div className="validation-header">
                  <span className="validation-icon">✓</span>
                  <strong>{validation.filename}</strong> — {validation.rows} rows, {validation.columns.length} columns
                </div>
                <div className="validation-cols">
                  Question column: <code>{validation.question_column}</code>
                  {validation.answer_column && <> · Answer column: <code>{validation.answer_column}</code></>}
                </div>
                {validation.warnings.length > 0 && (
                  <div className="validation-warnings">
                    {validation.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
                  </div>
                )}
                <table className="validation-preview">
                  <thead>
                    <tr><th>#</th><th>Question</th><th>Answer</th></tr>
                  </thead>
                  <tbody>
                    {validation.preview.map((r) => (
                      <tr key={r.index}>
                        <td>{r.index}</td>
                        <td>{r.question}</td>
                        <td>{r.answer}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            ) : (
              <div className="validation-header">
                <span className="validation-icon">✗</span>
                {validation.error}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Step 2: Experiment info */}
      <div className="card">
        <h2>2. Experiment</h2>
        <div className="toggle-row">
          <label className="toggle-switch">
            <input type="checkbox" checked={persist} onChange={(e) => setPersist(e.target.checked)} />
            <span className="toggle-slider"></span>
          </label>
          <span className="toggle-text">Save results to AWS (DynamoDB + S3)</span>
        </div>
        {persist && (
          <>
            <p className="hint" style={{ marginTop: 12 }}>
              Name this evaluation experiment. Results will be persisted so you can
              track and compare over time.
            </p>
            <div className="form-row">
              <div className="form-group">
                <label>Experiment Name <span className="optional-tag">optional</span></label>
                <input value={experimentName} onChange={(e) => setExperimentName(e.target.value)}
                  placeholder="e.g. Customer Support Agent v2" />
              </div>
              <div className="form-group">
                <label>Objective <span className="optional-tag">optional</span></label>
                <input value={experimentObjective} onChange={(e) => setExperimentObjective(e.target.value)}
                  placeholder="e.g. Measure quality after prompt changes" />
              </div>
            </div>
          </>
        )}
      </div>

      {/* Step 3: Agent connection */}
      <div className="card">
        <h2>3. Connect Your Agent</h2>
        <p className="hint">
          Select your agent framework and provide connection details. UAEF will
          send each ground truth question to your agent, collect the response
          trace, and evaluate it.
        </p>
        <div className="form-row">
          <div className="form-group">
            <label>Agent Framework</label>
            <select value={framework} onChange={(e) => setFramework(e.target.value)}>
              {agentTypes.map((t) => (
                <option key={t.id} value={t.id}>{t.label}</option>
              ))}
            </select>
          </div>
        </div>

        {/* AgentCore runtime picker */}
        {framework === "agentcore" && (
          <div className="form-row">
            <div className="form-group" style={{ flex: 2 }}>
              <label>Select Deployed Runtime</label>
              {loadingRuntimes ? (
                <p className="hint">Loading runtimes...</p>
              ) : agentCoreRuntimes.length > 0 ? (
                <select
                  value={connectionFields.agent_runtime_arn || ""}
                  onChange={(e) =>
                    setConnectionFields((prev) => ({
                      ...prev,
                      agent_runtime_arn: e.target.value,
                    }))
                  }
                >
                  <option value="">Select a runtime...</option>
                  {agentCoreRuntimes
                    .filter((r) => r.status === "READY")
                    .map((r) => (
                      <option key={r.arn} value={r.arn}>
                        {r.name} ({r.status})
                      </option>
                    ))}
                </select>
              ) : (
                <p className="hint">
                  No runtimes found. Enter the ARN manually below, or check your
                  AWS credentials and region.
                </p>
              )}
            </div>
          </div>
        )}

        {/* Dynamic connection fields based on selected framework */}
        {currentType && currentType.fields.length > 0 && (
          <div className="form-row">
            {currentType.fields.map((f) => (
              <div className="form-group" key={f.name}>
                <label>{f.label}{f.optional ? "" : " *"}</label>
                <input
                  value={connectionFields[f.name] || ""}
                  onChange={(e) =>
                    setConnectionFields((prev) => ({ ...prev, [f.name]: e.target.value }))
                  }
                  placeholder={f.placeholder || ""}
                />
              </div>
            ))}
          </div>
        )}

        {/* Help text for HTTP-based agents */}
        {currentType?.fields?.some((f) => f.name === "endpoint") && (
          <div className="info-box">
            Your agent endpoint should accept <code>POST</code> with{" "}
            <code>{`{"query": "..."}`}</code> and return its native response.
            The <strong>{framework}</strong> adapter will transform the raw
            output into a canonical trace for evaluation.
          </div>
        )}
      </div>

      {/* Step 4: Metrics */}
      <div className="card">
        <div className="card-header-row">
          <h2>
            4. Select Metrics ({selectedMetrics ? selectedMetrics.size : 0} of{" "}
            {allMetrics.length})
          </h2>
          <div className="btn-group">
            <button className="btn btn-outline btn-sm" onClick={() => setSelectedMetrics(new Set(allMetrics))}>All</button>
            <button className="btn btn-outline btn-sm" onClick={() => setSelectedMetrics(new Set())}>None</button>
          </div>
        </div>
        <div className="form-row" style={{ marginBottom: 8, gap: 8 }}>
          <div className="form-group" style={{ flex: "0 0 auto", minWidth: 180 }}>
            <label>Filter by GT requirement</label>
            <select value={metricGtFilter} onChange={(e) => setMetricGtFilter(e.target.value)}>
              <option value="all">All metrics</option>
              <option value="no_gt">No GT needed</option>
              <option value="gt">GT needed</option>
            </select>
          </div>
        </div>
        <div className="metrics-list">
          {Object.entries(metrics).map(([dim, mets]) => {
            const filteredMets = mets.filter((m) => {
              if (metricGtFilter === "no_gt") return !GT_REQUIRED_METRICS.has(m);
              if (metricGtFilter === "gt") return GT_REQUIRED_METRICS.has(m);
              return true;
            });
            if (filteredMets.length === 0) return null;
            const dimSelected = selectedMetrics ? filteredMets.filter((m) => selectedMetrics.has(m)).length : 0;
            const isExpanded = expandedDims.has(dim);
            return (
              <div key={dim} className="metric-dimension-row">
                <div className="dim-header">
                  <label className="dim-label">
                    <input
                      type="checkbox"
                      checked={dimSelected === filteredMets.length}
                      ref={(el) => { if (el) el.indeterminate = dimSelected > 0 && dimSelected < filteredMets.length; }}
                      onChange={() => {
                        setSelectedMetrics((prev) => {
                          const next = new Set(prev);
                          const allSelected = filteredMets.every((m) => next.has(m));
                          filteredMets.forEach((m) => (allSelected ? next.delete(m) : next.add(m)));
                          return next;
                        });
                      }}
                    />
                    <span>{dim}</span>
                    <span className="dim-count">({dimSelected}/{filteredMets.length})</span>
                  </label>
                  <button
                    className="btn-expand"
                    onClick={() => setExpandedDims((prev) => {
                      const next = new Set(prev);
                      next.has(dim) ? next.delete(dim) : next.add(dim);
                      return next;
                    })}
                  >
                    {isExpanded ? "▲" : "▼"}
                  </button>
                </div>
                {isExpanded && (
                  <div className="dim-metrics">
                    {filteredMets.map((m) => (
                      <label key={m} className="metric-label">
                        <input
                          type="checkbox"
                          checked={selectedMetrics ? selectedMetrics.has(m) : false}
                          onChange={() => toggleMetric(m)}
                        />
                        {m}
                        {GT_REQUIRED_METRICS.has(m) && <span className="tag tag-orange" style={{ marginLeft: 6, fontSize: "0.65rem" }}>GT</span>}
                      </label>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Run */}
      <div className="card">
        <button
          className="btn btn-primary btn-lg"
          onClick={handleEvaluate}
          disabled={loading || (dataSource === "upload" ? !file : !s3DataPath)}
        >
          {loading ? "⏳ Running evaluation..." : "🚀 Run Evaluation"}
        </button>
        {status && <p className="status-msg">{status}</p>}
        {error && <div className="error">{error}</div>}
      </div>
    </>
  );
}
