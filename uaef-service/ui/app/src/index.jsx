// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";
import { loadConfig } from "./config";
import { handleRedirectCallback, isAuthenticated, signIn } from "./auth";

async function bootstrap() {
  await loadConfig();
  // If returning from Hosted UI with ?code=..., complete the token exchange.
  try {
    await handleRedirectCallback();
  } catch (e) {
    console.error("Auth callback failed:", e);
  }

  const root = ReactDOM.createRoot(document.getElementById("root"));

  if (!isAuthenticated()) {
    // Simple sign-in gate before mounting the app.
    root.render(
      <div style={{ maxWidth: 520, margin: "12vh auto", textAlign: "center", fontFamily: "system-ui, sans-serif" }}>
        <h1>🔬 UAEF</h1>
        <p style={{ color: "#666" }}>Universal Agent Evaluation Framework</p>
        <p style={{ color: "#666", marginTop: 24 }}>Please sign in to continue.</p>
        <button
          onClick={() => signIn()}
          style={{ marginTop: 8, padding: "10px 20px", fontSize: 15, background: "#4f46e5", color: "#fff", border: 0, borderRadius: 8, cursor: "pointer" }}
        >
          Sign in
        </button>
      </div>
    );
    return;
  }

  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}

bootstrap();
