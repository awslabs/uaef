// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Runtime config injected by the CDK BucketDeployment as /config.json
// (region, userPoolId, clientId, cognitoDomain, apiBaseUrl, redirectUri).
// Loaded once at startup before the app renders.

let _cfg = null;

export async function loadConfig() {
  if (_cfg) return _cfg;
  const res = await fetch("config.json", { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load config.json (" + res.status + ")");
  _cfg = await res.json();
  return _cfg;
}

export function getConfig() {
  if (!_cfg) throw new Error("Config not loaded yet");
  return _cfg;
}
