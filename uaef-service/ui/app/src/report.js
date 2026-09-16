// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Open the library-generated report (uaef.reporting.generate_report) for an
// experiment in a new browser tab. The Worker returns a self-contained
// Markdown report (PNG charts inlined as base64); we render it to HTML with
// `marked` and write it into the opened window.

import { marked } from "marked";
import { api } from "./api";

const REPORT_CSS = `
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    line-height: 1.6; color: #1a1a1a; max-width: 960px; margin: 2rem auto; padding: 0 1.25rem; }
  h1 { border-bottom: 2px solid #eee; padding-bottom: .3rem; }
  h2 { color: #4f46e5; margin-top: 2rem; border-bottom: 1px solid #f0f0f0; padding-bottom: .25rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem; }
  th, td { border: 1px solid #e2e2e2; padding: 6px 10px; text-align: left; }
  th { background: #f7f7fb; }
  img { max-width: 100%; height: auto; display: block; margin: 1rem 0; }
  code { background: #f3f3f7; padding: 1px 4px; border-radius: 4px; }
  blockquote { border-left: 4px solid #ddd; margin: 1rem 0; padding: .25rem 1rem; color: #555; }
`;

export async function openExperimentReport(experimentId, reportType = "full") {
  // Open the tab synchronously (avoids popup blockers), then fill it in.
  const win = window.open("", "_blank");
  if (win) {
    win.document.write("<!doctype html><meta charset='utf-8'><title>Generating report…</title>" +
      "<p style='font-family:system-ui;margin:2rem'>Generating report… this can take a moment.</p>");
  }
  try {
    const { reportUrl, title } = await api.generateReport(experimentId, reportType);
    const md = await (await fetch(reportUrl)).text();
    const html = marked.parse(md);
    const doc = `<!doctype html><html><head><meta charset="utf-8">` +
      `<title>${title || "Report"}</title><style>${REPORT_CSS}</style></head>` +
      `<body>${html}</body></html>`;
    if (win) {
      win.document.open();
      win.document.write(doc);
      win.document.close();
    } else {
      // Popup blocked — fall back to a blob URL navigation.
      const blob = new Blob([doc], { type: "text/html" });
      window.location.assign(URL.createObjectURL(blob));
    }
  } catch (e) {
    const msg = "Failed to generate report: " + (e && e.message ? e.message : e);
    if (win) { win.document.open(); win.document.write(`<p style="font-family:system-ui;margin:2rem;color:#b91c1c">${msg}</p>`); win.document.close(); }
    else { alert(msg); }
  }
}
