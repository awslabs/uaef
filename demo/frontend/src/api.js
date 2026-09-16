// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

const BASE = "/api";

// Optional API key for the demo backend. Only sent when VITE_DEMO_API_KEY is
// defined at build time (matching the backend's UAEF_DEMO_API_KEY). The
// backend fails closed: if neither UAEF_DEMO_API_KEY nor
// UAEF_DEMO_ALLOW_NO_AUTH=true is set on it, every request is rejected — see
// demo/README.md.
const DEMO_API_KEY = import.meta.env?.VITE_DEMO_API_KEY || "";

async function request(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (DEMO_API_KEY) headers["X-API-Key"] = DEMO_API_KEY;
  const res = await fetch(`${BASE}${path}`, { ...opts, headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export const api = {
  getAwsIdentity: () => request("/aws-identity"),
  getMetrics: () => request("/metrics"),
  getAgentTypes: () => request("/agent-types"),
  getAgentCoreRuntimes: (region = "us-east-1") =>
    request(`/agentcore/runtimes?region=${encodeURIComponent(region)}`),
  getAgentCoreEnv: () => request("/agentcore/env"),
  evaluate: (formData) =>
    request("/evaluate", { method: "POST", body: formData }),
  getExperiments: () => request("/experiments"),
  compareExperiments: (ids) =>
    request("/compare-experiments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ experiment_ids: ids }),
    }),
  getExperimentResults: (id) =>
    request(`/experiments/${encodeURIComponent(id)}`),
};
