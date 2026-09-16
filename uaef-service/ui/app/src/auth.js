// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Cognito Hosted-UI login (OAuth authorization-code + PKCE) for the deployed UI.
// Mirrors the flow proven in the minimal UI: redirect to Hosted UI, exchange the
// code for an id_token (kept in sessionStorage), attach it as a Bearer token.

import { getConfig } from "./config";

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

export function getToken() {
  return sessionStorage.getItem("id_token");
}

export function isAuthenticated() {
  return !!getToken();
}

export function getUserEmail() {
  const t = getToken();
  if (!t) return null;
  try {
    const payload = JSON.parse(atob(t.split(".")[1]));
    return payload.email || payload["cognito:username"] || null;
  } catch {
    return null;
  }
}

export async function signIn() {
  const cfg = getConfig();
  const verifier = randomString(48);
  sessionStorage.setItem("pkce_verifier", verifier);
  const challenge = await challengeFrom(verifier);
  const u = new URL(cfg.cognitoDomain + "/login");
  u.searchParams.set("client_id", cfg.clientId);
  u.searchParams.set("response_type", "code");
  u.searchParams.set("scope", "openid email profile");
  u.searchParams.set("redirect_uri", cfg.redirectUri);
  u.searchParams.set("code_challenge_method", "S256");
  u.searchParams.set("code_challenge", challenge);
  window.location.assign(u.toString());
}

export function signOut() {
  sessionStorage.removeItem("id_token");
  const cfg = getConfig();
  const u = new URL(cfg.cognitoDomain + "/logout");
  u.searchParams.set("client_id", cfg.clientId);
  u.searchParams.set("logout_uri", cfg.redirectUri);
  window.location.assign(u.toString());
}

// If we're returning from Hosted UI with ?code=..., exchange it for tokens.
export async function handleRedirectCallback() {
  const cfg = getConfig();
  const params = new URLSearchParams(window.location.search);
  const code = params.get("code");
  if (!code) return;
  const verifier = sessionStorage.getItem("pkce_verifier") || "";
  const body = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: cfg.clientId,
    code,
    redirect_uri: cfg.redirectUri,
    code_verifier: verifier,
  });
  const res = await fetch(cfg.cognitoDomain + "/oauth2/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!res.ok) throw new Error("Token exchange failed: " + res.status);
  const data = await res.json();
  sessionStorage.setItem("id_token", data.id_token);
  window.history.replaceState({}, "", cfg.redirectUri);
}
