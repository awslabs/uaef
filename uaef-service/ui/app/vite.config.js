// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Deployed build: relative base so assets resolve under the CloudFront root.
// The app reads its backend URL + Cognito config from /config.json at runtime
// (no dev proxy in production).
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    port: 3000,
    host: true,
  },
});
