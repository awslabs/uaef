// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

// Runtime config injected by the CDK BucketDeployment (config.json).
let CFG = null;

const $ = (id) => document.getElementById(id);
const show = (id, on) => { $(id).style.display = on ? "" : "none"; };
const setStatus = (s) => { $("status").textContent = s; };
const setOut = (o) => { $("out").textContent = typeof o === "string" ? o : JSON.stringify(o, null, 2); };

function apiBase() { return CFG.apiBaseUrl.replace(/\/+$/, ""); }
function token() { return sessionStorage.getItem("id_token"); }

// ---- PKCE helpers ----
function randomString(n) {
  const a = new Uint8Array(n);
  crypto.getRandomValues(a);
  return Array.from(a, (b) => ("0" + (b & 0xff).toString(16)).slice(-2)).join("");
}
function base64url(buf) {
  return btoa(String.fromCharCode(...new Uint8Array(buf)))
    .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
async function challengeFrom(verifier) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
  return base64url(digest);
}

// ---- OAuth (Cognito Hosted UI, authorization code + PKCE) ----
async function signIn() {
  const verifier = randomString(48);
  sessionStorage.setItem("pkce_verifier", verifier);
  const challenge = await challengeFrom(verifier);
  const u = new URL(CFG.cognitoDomain + "/login");
  u.searchParams.set("client_id", CFG.clientId);
  u.searchParams.set("response_type", "code");
  u.searchParams.set("scope", "openid email profile");
  u.searchParams.set("redirect_uri", CFG.redirectUri);
  u.searchParams.set("code_challenge_method", "S256");
  u.searchParams.set("code_challenge", challenge);
  window.location.assign(u.toString());
}

function signOut() {
  sessionStorage.removeItem("id_token");
  const u = new URL(CFG.cognitoDomain + "/logout");
  u.searchParams.set("client_id", CFG.clientId);
  u.searchParams.set("logout_uri", CFG.redirectUri);
  window.location.assign(u.toString());
}

async function exchangeCode(code) {
  const verifier = sessionStorage.getItem("pkce_verifier") || "";
  const body = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: CFG.clientId,
    code,
    redirect_uri: CFG.redirectUri,
    code_verifier: verifier,
  });
  const res = await fetch(CFG.cognitoDomain + "/oauth2/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!res.ok) throw new Error("Token exchange failed: " + res.status);
  const data = await res.json();
  sessionStorage.setItem("id_token", data.id_token);
}

// ---- API calls (Bearer token; async job + poll) ----
async function authed(path, opts = {}) {
  const res = await fetch(apiBase() + path, {
    ...opts,
    headers: { Authorization: "Bearer " + token(), "Content-Type": "application/json", ...(opts.headers || {}) },
  });
  if (res.status === 401) {
    // Token missing/expired/invalid — the gateway returns a CORS-enabled 401,
    // so we can detect it here and send the user back through Hosted-UI login.
    sessionStorage.removeItem("id_token");
    setStatus("session expired — signing in again…");
    await signIn();
    throw new Error("Not authenticated (redirecting to login).");
  }
  const text = await res.text();
  let data; try { data = JSON.parse(text); } catch { data = text; }
  if (!res.ok) throw new Error("HTTP " + res.status + ": " + (data && data.error ? data.error : text));
  return data;
}

async function loadCatalog() {
  setStatus("loading catalog…");
  try { setOut(await authed("/metrics")); setStatus("idle"); }
  catch (e) { setStatus("error"); setOut(String(e)); }
}

async function run() {
  let trace;
  try { trace = JSON.parse($("trace").value); }
  catch { setStatus("error"); setOut("Trace must be valid JSON."); return; }
  const metrics = $("metrics").value.split(",").map((s) => s.trim()).filter(Boolean);

  setStatus("submitting…");
  try {
    const created = await authed("/evaluate", { method: "POST", body: JSON.stringify({ trace, metrics }) });
    const jobId = created.jobId;
    setStatus("job " + jobId + " — polling…");
    const deadline = Date.now() + 15 * 60 * 1000;
    while (Date.now() < deadline) {
      const job = await authed("/jobs/" + jobId);
      if (job.status === "COMPLETED") { setStatus("COMPLETED"); setOut(job); return; }
      if (job.status === "FAILED") { setStatus("FAILED"); setOut(job); return; }
      setStatus(job.status + " — polling…");
      await new Promise((r) => setTimeout(r, 2000));
    }
    setStatus("timeout");
  } catch (e) { setStatus("error"); setOut(String(e)); }
}

// ---- bootstrap ----
async function main() {
  CFG = await (await fetch("config.json")).json();
  $("signin").onclick = signIn;
  $("logout").onclick = signOut;
  $("run").onclick = run;
  $("catalog").onclick = loadCatalog;

  const params = new URLSearchParams(window.location.search);
  if (params.get("code")) {
    try {
      await exchangeCode(params.get("code"));
      window.history.replaceState({}, "", CFG.redirectUri);
    } catch (e) { setOut(String(e)); }
  }

  if (token()) { show("app", true); show("login", false); $("who").textContent = "signed in"; }
  else { show("app", false); show("login", true); }
}
main();
