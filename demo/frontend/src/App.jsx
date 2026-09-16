// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState, useEffect, useCallback } from "react";
import { api } from "./api";
import ExperimentsPanel from "./components/ExperimentsPanel";
import SetupPanel from "./components/SetupPanel";
import ResultsPanel from "./components/ResultsPanel";
import ComparePanel from "./components/ComparePanel";
import TaxonomyPanel from "./components/TaxonomyPanel";

const TABS = ["Experiments", "Evaluate", "Results", "Compare", "Taxonomy"];

export default function App() {
  const [tab, setTab] = useState("Experiments");
  const [metrics, setMetrics] = useState({});
  const [lastResult, setLastResult] = useState(null);
  const [awsIdentity, setAwsIdentity] = useState(null);

  const refreshIdentity = useCallback(() => {
    api.getAwsIdentity().then(setAwsIdentity).catch(() => {});
  }, []);

  useEffect(() => {
    api.getMetrics().then(setMetrics).catch(() => {});
    refreshIdentity();
  }, [refreshIdentity]);

  const handleEvaluationComplete = useCallback((result) => {
    setLastResult(result);
    setTab("Results");
  }, []);

  const handleViewExperiment = useCallback(async (experimentId) => {
    try {
      const result = await api.getExperimentResults(experimentId);
      setLastResult(result);
      setTab("Results");
    } catch (e) {
      console.error("Failed to load experiment results:", e);
    }
  }, []);

  return (
    <div className="app">
      <div className="header">
        <div className="header-left">
          <h1>🔬 UAEF</h1>
          <span>Universal Agent Evaluation Framework</span>
        </div>
        <div className="header-right">
          {awsIdentity && awsIdentity.authenticated ? (
            <button
              className="aws-badge aws-connected"
              onClick={refreshIdentity}
              style={{ cursor: "pointer", border: "none" }}
              title="Ambient AWS credentials — click to refresh"
            >
              <span className="aws-dot" />
              <div className="aws-info">
                <span className="aws-account">{awsIdentity.account}</span>
                <span className="aws-arn">{awsIdentity.arn.split("/").pop()}</span>
              </div>
            </button>
          ) : (
            <button
              className="aws-badge aws-disconnected"
              onClick={refreshIdentity}
              style={{ cursor: "pointer", border: "none" }}
              title="Set AWS credentials in your environment, then click to refresh"
            >
              <span className="aws-dot" />
              <span className="aws-label">AWS not connected — set credentials in your environment, then refresh</span>
            </button>
          )}
        </div>
      </div>

      <div className="tabs">
        {TABS.map((t) => (
          <button
            key={t}
            className={`tab ${tab === t ? "active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "Experiments" && <ExperimentsPanel onViewResults={handleViewExperiment} />}
      {tab === "Evaluate" && (
        <SetupPanel
          metrics={metrics}
          onComplete={handleEvaluationComplete}
        />
      )}
      {tab === "Results" && <ResultsPanel result={lastResult} />}
      {tab === "Compare" && <ComparePanel />}
      {tab === "Taxonomy" && <TaxonomyPanel />}
    </div>
  );
}
