// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState, useEffect } from "react";
import { api } from "../api";

// No client-side parsing: data files are uploaded to the service and parsed +
// validated server-side by the library (uaef.data). This panel only collects a
// data reference (uploaded payload key or an s3:// path) and a filename hint.
function basename(p) {
  return (p || "").replace(/\/+$/, "").split("/").pop() || "data.csv";
}

export default function SetupPanel({ metrics, onComplete }) {
  const [agentTypes, setAgentTypes] = useState([]);
  const [framework, setFramework] = useState("agentcore");
  const [connectionFields, setConnectionFields] = useState({});

  const [agentCoreRuntimes, setAgentCoreRuntimes] = useState([]);
  const [loadingRuntimes, setLoadingRuntimes] = useState(false);

  // Data: upload a file (CSV/XLSX) or reference an S3 path. The reference (a
  // payload key or s3:// URI) + filename are sent to the server, which parses.
  const [dataSource, setDataSource] = useState("upload");
  const [file, setFile] = useState(null);
  const [s3DataPath, setS3DataPath] = useState("");
  const [dataRef, setDataRef] = useState("");      // resolved reference for evaluation
  const [dataFilename, setDataFilename] = useState("");
  const [validation, setValidation] = useState(null);
  const [validating, setValidating] = useState(false);

  const [selectedMetrics, setSelectedMetrics] = useState(null);
  const [expandedDims, setExpandedDims] = useState(new Set());

  const [persist, setPersist] = useState(false);
  const [experimentName, setExperimentName] = useState("");
  const [experimentObjective, setExperimentObjective] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [progress, setProgress] = useState(null);

  useEffect(() => { api.getAgentTypes().then(setAgentTypes).catch(() => {}); }, []);

  useEffect(() => {
    const t = agentTypes.find((x) => x.id === framework);
    if (t) {
      const defaults = {};
      for (const f of t.fields) defaults[f.name] = f.default || "";
      setConnectionFields(defaults);
    }
  }, [framework, agentTypes]);

  useEffect(() => {
    if (framework !== "agentcore") return;
    setLoadingRuntimes(true);
    const region = connectionFields.region || "us-east-1";
    api.getAgentCoreRuntimes(region)
      .then((d) => setAgentCoreRuntimes(d.runtimes || []))
      .catch(() => setAgentCoreRuntimes([]))
      .finally(() => setLoadingRuntimes(false));
  }, [framework]); // eslint-disable-line react-hooks/exhaustive-deps

  // Default: select ALL metrics and expand all dimensions so the full catalog
  // is visible and chosen by default. Long runs are handled by the poll window
  // (just above the Worker's 15-min Lambda cap), not by trimming metrics.
  useEffect(() => {
    if (selectedMetrics === null && Object.keys(metrics).length > 0) {
      setSelectedMetrics(new Set(Object.values(metrics).flat()));
      setExpandedDims(new Set(Object.keys(metrics)));
    }
  }, [metrics, selectedMetrics]);

  const allMetrics = Object.values(metrics).flat();
  const currentType = agentTypes.find((t) => t.id === framework);
  const toggleMetric = (m) =>
    setSelectedMetrics((prev) => { const n = new Set(prev); n.has(m) ? n.delete(m) : n.add(m); return n; });

  const handleValidate = async () => {
    setValidating(true); setValidation(null); setError(""); setDataRef("");
    try {
      let ref, fname;
      if (dataSource === "upload") {
        if (!file) { setValidation({ valid: false, errors: ["Choose a file first."] }); return; }
        setStatus("Uploading file…");
        ref = await api.uploadData(file);   // payload key
        fname = file.name;
      } else {
        if (!s3DataPath.startsWith("s3://")) { setValidation({ valid: false, errors: ["Path must start with s3://"] }); return; }
        ref = s3DataPath.trim();
        fname = basename(s3DataPath);
      }
      setStatus("Validating…");
      const result = await api.validateData(ref, fname);  // server parses via uaef.data
      setValidation(result);
      if (result.valid) { setDataRef(ref); setDataFilename(fname); }
      setStatus("");
    } catch (e) {
      setValidation({ valid: false, errors: [e.message] });
      setStatus("");
    } finally {
      setValidating(false);
    }
  };

  const handleEvaluate = async () => {
    setError(""); setStatus("");
    for (const f of (currentType?.fields || [])) {
      if (!f.optional && !connectionFields[f.name]) { setError(`${f.label} is required`); return; }
    }
    if (framework === "agentcore" && !connectionFields.agent_runtime_arn) {
      setError("Select or enter an AgentCore runtime ARN"); return;
    }

    setLoading(true);
    // Resolve the data reference. Validation is OPTIONAL — if the user didn't
    // click Validate, upload the file (or use the S3 path) right here.
    let ref = dataRef;
    let fname = dataFilename;
    try {
      if (dataSource === "upload") {
        if (!ref) {
          if (!file) { setError("Choose a CSV/XLSX file."); setLoading(false); return; }
          setStatus("Uploading file…");
          ref = await api.uploadData(file);
          fname = file.name;
        }
      } else {
        if (!s3DataPath.startsWith("s3://")) { setError("Enter a valid s3:// path."); setLoading(false); return; }
        ref = s3DataPath.trim();
        fname = basename(s3DataPath);
      }
    } catch (e) {
      setError(e.message); setStatus(""); setLoading(false); return;
    }

    const agentName =
      framework === "agentcore" && connectionFields.agent_runtime_arn
        ? (agentCoreRuntimes.find((r) => r.arn === connectionFields.agent_runtime_arn)?.name || connectionFields.agent_runtime_arn)
        : (currentType?.label || framework);

    const payload = {
      framework,
      connection: connectionFields,
      s3_data_path: ref,
      filename: fname,
      metrics: selectedMetrics ? [...selectedMetrics] : [],
      persist,
      agent_name: agentName,
    };
    if (persist && experimentName) payload.experiment_name = experimentName;
    if (persist && experimentObjective) payload.experiment_objective = experimentObjective;

    setStatus("Starting evaluation…");
    try {
      const result = await api.invokeEvaluate(payload, {
        onProgress: (p) => { setProgress(p); setStatus(""); },
      });
      setStatus("");
      setProgress(null);
      onComplete(result);
    } catch (e) {
      setError(e.message); setStatus(""); setProgress(null);
    }
    setLoading(false);
  };

  return (
    <>
      {/* Step 1: agent connection */}
      <div className="card">
        <h2>1. Connect Your Agent</h2>
        <p className="hint">
          The deployed service invokes agents reachable from AWS (AgentCore
          runtimes, Bedrock agents, public HTTPS endpoints). For a local agent,
          use the notebook instead.
        </p>
        <div className="form-row">
          <div className="form-group">
            <label>Agent Framework</label>
            <select value={framework} onChange={(e) => setFramework(e.target.value)}>
              {agentTypes.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
            </select>
          </div>
        </div>

        {framework === "agentcore" && (
          <div className="form-row">
            <div className="form-group" style={{ flex: 2 }}>
              <label>Deployed Runtime</label>
              {loadingRuntimes ? (
                <p className="hint">Loading runtimes…</p>
              ) : agentCoreRuntimes.length > 0 ? (
                <select
                  value={connectionFields.agent_runtime_arn || ""}
                  onChange={(e) => setConnectionFields((p) => ({ ...p, agent_runtime_arn: e.target.value }))}
                >
                  <option value="">Select a runtime…</option>
                  {agentCoreRuntimes.filter((r) => r.status === "READY").map((r) => (
                    <option key={r.arn} value={r.arn}>{r.name}</option>
                  ))}
                </select>
              ) : (
                <p className="hint">No runtimes found — enter an ARN below.</p>
              )}
            </div>
          </div>
        )}

        {currentType && currentType.fields.length > 0 && (
          <div className="form-row">
            {currentType.fields.map((f) => (
              <div className="form-group" key={f.name}>
                <label>{f.label}{f.optional ? "" : " *"}</label>
                <input
                  value={connectionFields[f.name] || ""}
                  onChange={(e) => setConnectionFields((p) => ({ ...p, [f.name]: e.target.value }))}
                  placeholder={f.placeholder || ""}
                />
              </div>
            ))}
          </div>
        )}

      </div>

      {/* Step 2: data */}
      <div className="card">
        <h2>2. Upload Evaluation Data</h2>
        <p className="hint">
          Provide test data (questions + optional expected answers). Upload a
          CSV/XLSX or point to an S3 path — the service parses and validates it.
          Columns: <code>question</code>, optional <code>answer</code>,
          <code>context</code>, <code>expected_tool_calls</code>.
        </p>
        <div className="form-row" style={{ alignItems: "end" }}>
          <div className="form-group" style={{ flex: "0 0 auto", minWidth: 160 }}>
            <label>Source</label>
            <select value={dataSource} onChange={(e) => { setDataSource(e.target.value); setValidation(null); setDataRef(""); }}>
              <option value="upload">Upload File</option>
              <option value="s3">S3 Path</option>
            </select>
          </div>
          {dataSource === "upload" ? (
            <div className="form-group" style={{ flex: 2 }}>
              <label>Ground Truth File (.csv, .xlsx)</label>
              <input type="file" accept=".csv,.xlsx,.xls"
                onChange={(e) => { setFile(e.target.files[0]); setValidation(null); setDataRef(""); }} />
            </div>
          ) : (
            <div className="form-group" style={{ flex: 2 }}>
              <label>S3 Path</label>
              <input value={s3DataPath} onChange={(e) => { setS3DataPath(e.target.value); setValidation(null); setDataRef(""); }}
                placeholder="s3://my-bucket/data/ground-truth.xlsx" />
            </div>
          )}
          <button className="btn btn-outline" style={{ alignSelf: "flex-end" }}
            disabled={validating || (dataSource === "upload" ? !file : !s3DataPath)}
            onClick={handleValidate}>
            {validating ? "Validating…" : "Validate"}
          </button>
        </div>

        {validation && (
          <div className={`validation-result ${validation.valid ? "validation-ok" : "validation-err"}`}>
            {validation.valid ? (
              <>
                <div className="validation-header">
                  <span className="validation-icon">✓</span>
                  <strong>{validation.filename || dataFilename}</strong> — {validation.rows} rows
                  {validation.question_column && <> · question: <code>{validation.question_column}</code></>}
                  {validation.answer_column && <> · answer: <code>{validation.answer_column}</code></>}
                </div>
                {/* Conversation shape, from the same grouping the run uses. Nothing
                    else in the UI reveals whether a file is multi-turn, and the
                    difference decides both which metrics apply and whether the
                    endpoint needs its own session store. */}
                {validation.multi_turn ? (
                  <div className="validation-cols">
                    <strong>MULTI-TURN</strong> — {validation.rows} rows,{" "}
                    {validation.session_count} conversations (grouped by <code>session_id</code>).
                    Your endpoint must keep conversation state and key on the{" "}
                    <code>session_id</code> sent each turn; prior turns are not replayed.
                  </div>
                ) : (
                  <div className="validation-cols">
                    <strong>SINGLE-TURN</strong> — {validation.rows} rows, each evaluated
                    independently.
                  </div>
                )}
                {validation.columns && <div className="validation-cols">Columns: {validation.columns.join(", ")}</div>}
                {(validation.warnings || []).length > 0 && (
                  <div className="validation-warnings">
                    {validation.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
                  </div>
                )}
                {(validation.preview || []).length > 0 && (
                  <table className="validation-preview">
                    <thead><tr><th>#</th><th>Question</th><th>Answer</th></tr></thead>
                    <tbody>
                      {validation.preview.map((r) => (
                        <tr key={r.index}><td>{r.index}</td><td>{r.question}</td><td>{r.answer || "—"}</td></tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </>
            ) : (
              <div className="validation-header">
                <span className="validation-icon">✗</span>
                {(validation.errors || [validation.error]).filter(Boolean).join("; ") || "Validation failed"}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Step 3: experiment */}
      <div className="card">
        <h2>3. Experiment</h2>
        <div className="toggle-row">
          <label className="toggle-switch">
            <input type="checkbox" checked={persist} onChange={(e) => setPersist(e.target.checked)} />
            <span className="toggle-slider"></span>
          </label>
          <span className="toggle-text">Name &amp; save this experiment</span>
        </div>
        <p className="hint" style={{ marginTop: 6 }}>
          Every run is persisted (DynamoDB + S3) so results and aggregates come
          from the library. Toggle on to give it a name/objective.
        </p>
        {persist && (
          <div className="form-row" style={{ marginTop: 12 }}>
            <div className="form-group">
              <label>Experiment Name <span className="optional-tag">optional</span></label>
              <input value={experimentName} onChange={(e) => setExperimentName(e.target.value)} placeholder="e.g. QnA Agent v3" />
            </div>
            <div className="form-group">
              <label>Objective <span className="optional-tag">optional</span></label>
              <input value={experimentObjective} onChange={(e) => setExperimentObjective(e.target.value)} placeholder="e.g. Baseline accuracy" />
            </div>
          </div>
        )}
      </div>

      {/* Step 4: metrics */}
      <div className="card">
        <div className="card-header-row">
          <h2>4. Select Metrics ({selectedMetrics ? selectedMetrics.size : 0} of {allMetrics.length})</h2>
          <div className="btn-group">
            <button className="btn btn-outline btn-sm" onClick={() => setSelectedMetrics(new Set(allMetrics))}>All</button>
            <button className="btn btn-outline btn-sm" onClick={() => setSelectedMetrics(new Set())}>None</button>
          </div>
        </div>
        {allMetrics.length === 0 && <p className="hint">Loading metric catalog…</p>}
        <div className="metrics-list">
          {Object.entries(metrics).map(([dim, mets]) => {
            const isExpanded = expandedDims.has(dim);
            const dimSelected = selectedMetrics ? mets.filter((m) => selectedMetrics.has(m)).length : 0;
            return (
              <div key={dim} className="metric-dimension-row">
                <div className="dim-header">
                  <label className="dim-label">
                    <input
                      type="checkbox"
                      checked={dimSelected === mets.length && mets.length > 0}
                      ref={(el) => { if (el) el.indeterminate = dimSelected > 0 && dimSelected < mets.length; }}
                      onChange={() => setSelectedMetrics((prev) => {
                        const next = new Set(prev);
                        const all = mets.every((m) => next.has(m));
                        mets.forEach((m) => (all ? next.delete(m) : next.add(m)));
                        return next;
                      })}
                    />
                    <span>{dim}</span>
                    <span className="dim-count">({dimSelected}/{mets.length})</span>
                  </label>
                  <button className="btn-expand" onClick={() => setExpandedDims((prev) => {
                    const next = new Set(prev); next.has(dim) ? next.delete(dim) : next.add(dim); return next;
                  })}>{isExpanded ? "▲" : "▼"}</button>
                </div>
                {isExpanded && (
                  <div className="dim-metrics">
                    {mets.map((m) => (
                      <label key={m} className="metric-label">
                        <input type="checkbox" checked={selectedMetrics ? selectedMetrics.has(m) : false} onChange={() => toggleMetric(m)} />
                        {m}
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
        <button className="btn btn-primary btn-lg" onClick={handleEvaluate} disabled={loading}>
          {loading ? "⏳ Running…" : "🚀 Run Evaluation"}
        </button>
        {loading && progress && progress.dimensions && (
          <div className="eval-progress">
            <div className="eval-progress-header">
              <span>Evaluating selected dimensions ({Object.values(progress.dimensions).filter(s => s === "done").length}/{progress.total || Object.keys(progress.dimensions).length})</span>
            </div>
            <div className="eval-progress-bar">
              <div className="eval-progress-fill" style={{
                width: `${(Object.values(progress.dimensions).filter(s => s === "done").length / (progress.total || Object.keys(progress.dimensions).length)) * 100}%`
              }} />
            </div>
            {(() => {
              const done = Object.entries(progress.dimensions).filter(([, s]) => s === "done").map(([d]) => d);
              if (done.length > 0) return <div className="eval-progress-done">✓ Completed evaluation for {done.join(", ")}</div>;
              return null;
            })()}
            <div className="eval-progress-status">
              {(() => {
                const running = Object.entries(progress.dimensions).filter(([, s]) => s === "running").map(([d]) => d);
                if (running.length > 0) return `Now running evaluation on ${running.join(", ")} metrics`;
                const pending = Object.entries(progress.dimensions).filter(([, s]) => s === "pending").map(([d]) => d);
                if (pending.length > 0) return `Waiting to evaluate ${pending[0]} metrics…`;
                return "Finalizing results…";
              })()}
            </div>
          </div>
        )}
        {status && <p className="status-msg">{status}</p>}
        {error && <div className="error">{error}</div>}
      </div>
    </>
  );
}
