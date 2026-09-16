// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// API client for the DEPLOYED UAEF service (API Gateway + Cognito + async jobs).
//
// Differs from the original demo client: every call carries a Cognito Bearer
// token, and evaluation is asynchronous (POST returns a jobId, then we poll
// GET /jobs/{id} and fetch the full result from the presigned resultRef).

import { getConfig } from "./config";
import { getToken, signIn } from "./auth";

function apiBase() {
  return getConfig().apiBaseUrl.replace(/\/+$/, "");
}

// fetch() throws a TypeError ("Failed to fetch") for network/CORS/DNS/mixed-
// content failures and tells JS nothing else (by design, for cross-origin
// safety). To make these debuggable, wrap every fetch so the thrown error names
// the method + URL that actually died, and log it to the console.
async function fetchOrThrow(url, opts = {}, label = "") {
  const method = (opts.method || "GET").toUpperCase();
  const tag = label ? `${label} ` : "";
  try {
    return await fetch(url, opts);
  } catch (e) {
    // This branch = the request never completed (CORS preflight blocked, DNS,
    // offline, mixed content, redirect-on-preflight, cert). NOT an HTTP error.
    const detail = `${tag}network/CORS failure: ${method} ${url}`;
    // nosemgrep: javascript.lang.security.audit.unsafe-formatstring.unsafe-formatstring -- False positive: browser console.error of an internal diagnostic string; `detail` is built from our own labels, not a user-controlled format specifier.
    console.error("[api] " + detail, e);
    const err = new Error(`Failed to fetch — ${detail}`);
    err.cause = e;
    err.kind = "network";
    err.url = url;
    throw err;
  }
}

async function request(path, opts = {}) {
  const url = apiBase() + path;
  const res = await fetchOrThrow(url, {
    ...opts,
    headers: {
      Authorization: "Bearer " + getToken(),
      "Content-Type": "application/json",
      ...(opts.headers || {}),
    },
  }, "[API]");
  if (res.status === 401) {
    // Token missing/expired — send the user back through Hosted-UI login.
    sessionStorage.removeItem("id_token");
    await signIn();
    throw new Error("Session expired — redirecting to login.");
  }
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = text; }
  if (!res.ok) {
    const msg = (data && data.error) || text || "HTTP " + res.status;
    throw new Error(msg);
  }
  return data;
}

// Poll a bit beyond the Worker Lambda's 15-min hard limit so a result that
// lands right at the limit is still caught. The Worker cannot run longer than
// 15 min (Lambda max), so polling past ~16 min only matters for catching that
// final terminal write, not for letting the job run longer.
async function pollJob(jobId, { interval = 3000, timeout = 16 * 60 * 1000, onProgress } = {}) {
  const deadline = Date.now() + timeout;
  // A long evaluation produces hundreds of poll requests. A single transient
  // network blip (corporate-proxy connection reset, momentary DNS/TLS hiccup)
  // throws "Failed to fetch" on ONE poll — that must NOT abort the whole run,
  // because the job keeps executing server-side. Tolerate transient failures
  // and keep polling until the deadline; only give up on a real FAILED status,
  // a non-network error (e.g. session expiry), or sustained connection loss.
  let consecutiveErrors = 0;
  const MAX_CONSECUTIVE_ERRORS = 10; // ~30s of continuous outage before bailing
  while (Date.now() < deadline) {
    let job;
    try {
      job = await request("/jobs/" + encodeURIComponent(jobId));
      consecutiveErrors = 0;
    } catch (e) {
      // Non-network errors (real HTTP failure, session expiry) are terminal.
      if (e.kind !== "network") throw e;
      consecutiveErrors += 1;
      console.warn(
        `[api] poll ${jobId} transient failure ` +
        `${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}: ${e.message}`
      );
      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        throw new Error(
          "Lost connection while polling the evaluation " +
          `(${consecutiveErrors} consecutive network failures). The job may ` +
          "still be running — check the Experiments tab for the result."
        );
      }
      await new Promise((r) => setTimeout(r, interval));
      continue;
    }
    if (job.status === "COMPLETED") return job;
    if (job.status === "FAILED") {
      throw new Error((job.error && job.error.message) || "Evaluation failed");
    }
    if (onProgress && job.progress) {
      onProgress(job.progress);
    }
    await new Promise((r) => setTimeout(r, interval));
  }
  throw new Error("Evaluation timed out");
}

async function fetchResult(job) {
  // resultRef is a presigned GET URL (no auth header) to the UI-shaped result.
  if (job.resultRef) {
    const r = await fetch(job.resultRef);
    if (r.ok) return r.json();
  }
  return job.resultSummary || {};
}

export const api = {
  // --- catalog / config ---
  getMetrics: () => request("/metrics"),
  getAgentTypes: () => request("/agent-types"),
  getAgentCoreRuntimes: (region = "us-east-1") =>
    request("/agentcore/runtimes?region=" + encodeURIComponent(region)),
  // No deployed endpoint for local .env discovery; the deployed UI drives
  // AWS-reachable agents only.
  getAgentCoreEnv: async () => ({ found: false }),

  // --- ground-truth data (uploaded to S3; parsed/validated server-side via uaef.data) ---
  // Upload a file via the presigned /payloads flow; returns the object key used
  // as the data reference. No client-side parsing — the library parses it.
  uploadData: async (file) => {
    const presign = await request("/payloads", { method: "POST" });
    // Log the exact presigned host so a "Failed to fetch" here is traceable to
    // the S3 endpoint (region/redirect/CORS) rather than the API.
    try {
      console.info("[api] presigned PUT host:", new URL(presign.uploadUrl).host);
    } catch { /* ignore */ }
    // No explicit Content-Type header: the presigned URL doesn't sign one, so
    // sending a mismatched header can cause S3 to reject the PUT with 403
    // (SignatureDoesNotMatch). Let the browser set it from the File.
    const put = await fetchOrThrow(presign.uploadUrl, { method: "PUT", body: file }, "[S3-PUT]");
    if (!put.ok) {
      const detail = await put.text().catch(() => "");
      throw new Error("Upload failed: " + put.status + (detail ? " — " + detail.slice(0, 300) : ""));
    }
    return presign.key;
  },
  validateData: (dataRef, filename) =>
    request("/validate-data", {
      method: "POST",
      body: JSON.stringify({ data_ref: dataRef, filename }),
    }),

  // --- evaluate (async: submit -> poll -> fetch result) ---
  invokeEvaluate: async (payload, { onProgress } = {}) => {
    const created = await request("/invoke-evaluate", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const job = await pollJob(created.jobId, { onProgress });
    return fetchResult(job);
  },

  // --- library report (uaef.reporting.generate_report via the Worker) ---
  // Returns { reportUrl, title } — reportUrl is a presigned GET to the
  // self-contained Markdown report (images inlined as base64).
  generateReport: (experimentId, reportType = "full") =>
    request("/reports", {
      method: "POST",
      body: JSON.stringify({ experiment_id: experimentId, report_type: reportType }),
    }),

  // --- experiments (best-effort; may be unavailable in the deployed service) ---
  getExperiments: () => request("/experiments"),
  getExperimentResults: (id) => request("/experiments/" + encodeURIComponent(id)),
  compareExperiments: (experimentIds) =>
    request("/compare-experiments", {
      method: "POST",
      body: JSON.stringify({ experiment_ids: experimentIds }),
    }),
};
