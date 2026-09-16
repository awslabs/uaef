# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reporting pipeline — single entry point for generating reports from evaluation results.

Provides `generate_report()` for single-run analysis and `generate_comparison_report()`
for multi-run comparison with regression detection.

Output structure:
    output/reports/<report_name>/
        report.md          — Main markdown report with embedded image links
        images/            — Generated plot images (PNG)

Examples:
    >>> from uaef.reporting.pipeline import generate_report, generate_comparison_report
    >>> from uaef.utils import load_metric_results
    >>>
    >>> results = load_metric_results("output/evaluation-results/run.json")
    >>> report_dir = generate_report(results)
    >>>
    >>> runs = [load_metric_results(p) for p in paths]
    >>> report_dir = generate_comparison_report(runs, labels=["Day 1", "Day 2", "Day 3"])
"""

import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from uaef.logging import get_logger
from uaef.models.evaluation_result import EvaluationResult
from uaef.utils.constants import REPORTS_OUTPUT_DIR

logger = get_logger(__name__)


# =============================================================================
# Public API
# =============================================================================


def generate_report(
    results: Union[EvaluationResult, List[EvaluationResult]],
    *,
    title: Optional[str] = None,
    report_type: str = "full",
    output_dir: str = REPORTS_OUTPUT_DIR,
    report_name: Optional[str] = None,
    include_recommendations: bool = True,
) -> str:
    """
    Generate a single-run evaluation report.

    Sections included:
    - Executive summary with pass/fail badge
    - Dimension breakdown table + radar chart
    - Per-metric scores table
    - Statistical outlier analysis with 95% confidence intervals (full only)
    - Score distribution histograms (full only)
    - Metric correlation heatmap (full only)
    - Safety summary (full only)
    - Failures list
    - Recommendations

    Args:
        results: EvaluationResult or list of results from a single evaluation run
        title: Report title (auto-generated if not provided)
        report_type: "full" (all sections) or "executive" (summary only)
        output_dir: Base directory for report output
        report_name: Folder name (auto-generated with timestamp if None)
        include_recommendations: Whether to include the recommendations section

    Returns:
        Path to the report directory containing report.md and images/
    """
    if isinstance(results, EvaluationResult):
        results = [results]
    if not results:
        raise ValueError("Cannot generate report from empty results")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if report_name is None:
        report_name = f"{report_type}_report_{timestamp}"

    report_dir = Path(output_dir) / report_name
    images_dir = report_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = f"{report_type.title()} Report — {len(results)} test case(s)"

    logger.info(f"Generating {report_type} report for {len(results)} results")

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append("")

    # --- Executive Summary ---
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("High-level overview of agent performance across all test cases.")
    lines.append("")
    lines.extend(_build_executive_summary(results))
    lines.append("")

    # --- Dimension Breakdown ---
    lines.append("## Dimension Breakdown")
    lines.append("")
    lines.append("Aggregate scores per evaluation dimension. Each dimension groups related metrics.")
    lines.append("")
    lines.extend(_build_dimension_table(results))
    lines.append("")
    radar_path = _generate_radar_chart(results, images_dir)
    if radar_path:
        lines.append(f"![Dimension Radar](images/{radar_path.name})")
        lines.append("")

    # --- Per-Metric Scores ---
    lines.append("## Per-Metric Scores")
    lines.append("")
    lines.append("Individual metric scores across all test cases.")
    lines.append("")
    lines.extend(_build_metric_table(results))
    lines.append("")

    if report_type == "full":
        # --- Statistical Outliers ---
        lines.append("## Statistical Outlier Analysis")
        lines.append("")
        lines.append("95% confidence intervals per metric. Outliers are scores beyond 2 standard deviations from the mean.")
        lines.append("")
        lines.extend(_build_outlier_section(results, images_dir))
        lines.append("")

        # --- Score Distributions ---
        lines.append("## Score Distributions")
        lines.append("")
        lines.append("Histograms showing the distribution shape of metric scores. Metrics with highest variance are shown.")
        lines.append("")
        hist_path = _generate_histogram(results, images_dir)
        if hist_path:
            lines.append(f"![Score Distributions](images/{hist_path.name})")
        else:
            lines.append("*Insufficient data for histograms (need ≥5 test cases).*")
        lines.append("")

        # --- Correlation Heatmap ---
        lines.append("## Metric Correlations")
        lines.append("")
        lines.append("Pearson correlation between metrics. High correlation indicates metrics that tend to move together.")
        lines.append("")
        corr_path = _generate_correlation_heatmap(results, images_dir)
        if corr_path:
            lines.append(f"![Correlation Heatmap](images/{corr_path.name})")
        else:
            lines.append("*Insufficient data for correlation analysis.*")
        lines.append("")

        # --- Safety Summary ---
        lines.append("## Safety Summary")
        lines.append("")
        lines.append("Responsible AI metrics overview. Concerns are flagged when scores drop below 0.7.")
        lines.append("")
        lines.extend(_build_safety_summary(results))
        lines.append("")

    # --- Failures ---
    failures_section = _build_failures_section(results)
    if failures_section:
        lines.append("## Failures")
        lines.append("")
        lines.append("Test cases that failed threshold checks.")
        lines.append("")
        lines.extend(failures_section)
        lines.append("")

    # --- Recommendations ---
    if include_recommendations:
        lines.append("## Recommendations")
        lines.append("")
        lines.append("Actionable suggestions based on dimension scores and metric variance.")
        lines.append("")
        lines.extend(_build_recommendations(results))
        lines.append("")

    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved to {report_dir}")
    return str(report_dir)


def generate_comparison_report(
    runs: List[List[EvaluationResult]],
    *,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    output_dir: str = REPORTS_OUTPUT_DIR,
    report_name: Optional[str] = None,
) -> str:
    """
    Generate a multi-run comparison report with regression detection.

    Sections included:
    - Overall scores comparison table
    - Side-by-side metric comparison chart
    - Percentage change chart (vs baseline)
    - Full delta table (every metric across all runs)
    - Metric trend lines chart
    - Regressions & improvements summary (>5% threshold)
    - Dimension comparison radar overlay

    Args:
        runs: List of result lists, one per run (ordered chronologically)
        labels: Human-readable label for each run
        title: Report title
        output_dir: Base directory for report output
        report_name: Folder name (auto-generated if None)

    Returns:
        Path to the report directory containing report.md and images/
    """
    if len(runs) < 2:
        raise ValueError("Need at least 2 runs to compare")

    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(runs))]
    if len(labels) != len(runs):
        raise ValueError("labels length must match runs length")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if report_name is None:
        report_name = f"comparison_report_{timestamp}"
    if title is None:
        title = f"Comparison Report — {len(runs)} runs"

    report_dir = Path(output_dir) / report_name
    images_dir = report_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating comparison report for {len(runs)} runs")

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append(f"*Comparing {len(runs)} runs: {', '.join(labels)}*")
    lines.append("")

    # --- Overall Scores ---
    lines.append("## Overall Scores")
    lines.append("")
    lines.append("Summary statistics for each evaluation run.")
    lines.append("")
    lines.append("| Run | Avg Score | Pass Rate | Test Cases |")
    lines.append("|-----|-----------|-----------|------------|")
    for run, label in zip(runs, labels):
        avg = statistics.mean([r.overall_score for r in run])
        pr = sum(1 for r in run if r.passed) / len(run) * 100
        lines.append(f"| {label} | {avg:.3f} | {pr:.0f}% | {len(run)} |")
    lines.append("")

    # --- Side-by-Side Chart ---
    lines.append("## Side-by-Side Comparison")
    lines.append("")
    lines.append("Average metric scores for each run displayed side by side.")
    lines.append("")
    sidebyside_path = _generate_side_by_side_chart(runs, labels, images_dir)
    if sidebyside_path:
        lines.append(f"![Side-by-Side Comparison](images/{sidebyside_path.name})")
    lines.append("")

    # --- Percentage Change ---
    lines.append("## Percentage Change from Baseline")
    lines.append("")
    lines.append(f"Percentage change of each metric relative to *{labels[0]}*. Negative values indicate regression.")
    lines.append("")
    pct_path = _generate_percentage_change_chart(runs, labels, images_dir)
    if pct_path:
        lines.append(f"![Percentage Change](images/{pct_path.name})")
    lines.append("")

    # --- Delta Table ---
    lines.append("## Detailed Changes")
    lines.append("")
    lines.append(f"Per-metric averages across all runs with absolute delta (last run vs baseline).")
    lines.append("")
    lines.extend(_build_delta_table(runs, labels))
    lines.append("")

    # --- Metric Trends ---
    lines.append("## Metric Trends")
    lines.append("")
    lines.append("Line chart showing how the top changing metrics evolve across runs.")
    lines.append("")
    trend_path = _generate_trend_chart(runs, labels, images_dir)
    if trend_path:
        lines.append(f"![Metric Trends](images/{trend_path.name})")
    lines.append("")

    # --- Regressions & Improvements ---
    lines.append("## Regressions & Improvements")
    lines.append("")
    lines.append("Metrics that changed by more than 5% between baseline and the latest run.")
    lines.append("")
    lines.extend(_build_metric_trend_table(runs, labels))
    lines.append("")

    # --- Dimension Comparison Radar ---
    lines.append("## Dimension Comparison")
    lines.append("")
    lines.append("Radar overlay comparing dimension-level performance across all runs.")
    lines.append("")
    radar_path = _generate_comparison_radar(runs, labels, images_dir)
    if radar_path:
        lines.append(f"![Dimension Comparison](images/{radar_path.name})")
    lines.append("")

    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Comparison report saved to {report_dir}")
    return str(report_dir)


# =============================================================================
# Section Builders — Single Run
# =============================================================================


def _build_executive_summary(results: List[EvaluationResult]) -> List[str]:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_score = statistics.mean([r.overall_score for r in results])
    pass_rate = passed / total * 100

    if avg_score >= 0.9:
        badge = "✅ PASS"
    elif avg_score >= 0.7:
        badge = "⚠️ WARNING"
    else:
        badge = "❌ FAIL"

    return [
        f"**Status: {badge}** (avg: {avg_score:.2f})",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total test cases | {total} |",
        f"| Average score | {avg_score:.3f} |",
        f"| Pass rate | {pass_rate:.0f}% ({passed}/{total}) |",
        f"| Failed | {total - passed} |",
    ]


def _build_dimension_table(results: List[EvaluationResult]) -> List[str]:
    dim_scores = _collect_dimension_scores(results)
    if not dim_scores:
        return ["No dimension data available."]
    lines = [
        "| Dimension | Avg Score | Min | Max | Std Dev |",
        "|-----------|-----------|-----|-----|---------|",
    ]
    for dim_name, scores in sorted(dim_scores.items()):
        avg = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        lines.append(f"| {dim_name} | {avg:.3f} | {min(scores):.3f} | {max(scores):.3f} | {std:.3f} |")
    return lines


def _build_metric_table(results: List[EvaluationResult]) -> List[str]:
    metric_scores = _collect_metric_scores(results)
    if not metric_scores:
        return ["No metric data available."]
    lines = [
        "| Metric | Avg | Min | Max | N |",
        "|--------|-----|-----|-----|---|",
    ]
    for name, scores in sorted(metric_scores.items()):
        avg = statistics.mean(scores)
        lines.append(f"| {name} | {avg:.3f} | {min(scores):.3f} | {max(scores):.3f} | {len(scores)} |")
    return lines


def _build_outlier_section(results: List[EvaluationResult], images_dir: Path) -> List[str]:
    metric_scores = _collect_metric_scores(results)
    if not metric_scores:
        return ["Insufficient data for outlier analysis."]

    lines = ["| Metric | Mean | 95% CI Lower | 95% CI Upper | Outliers |",
             "|--------|------|--------------|--------------|----------|"]
    outlier_data = []
    for name, scores in sorted(metric_scores.items()):
        if len(scores) < 3:
            continue
        mean = statistics.mean(scores)
        std = statistics.stdev(scores)
        n = len(scores)
        margin = 1.96 * std / (n ** 0.5)
        ci_lower = max(0.0, mean - margin)
        ci_upper = min(1.0, mean + margin)
        outliers = [s for s in scores if s < mean - 2 * std or s > mean + 2 * std]
        lines.append(f"| {name} | {mean:.3f} | {ci_lower:.3f} | {ci_upper:.3f} | {len(outliers)} |")
        outlier_data.append((name, mean, ci_lower, ci_upper, scores))

    if outlier_data:
        lines.append("")
        plot_path = _generate_ci_plot(outlier_data, images_dir)
        if plot_path:
            lines.append(f"![Confidence Intervals](images/{plot_path.name})")
    return lines


def _build_safety_summary(results: List[EvaluationResult]) -> List[str]:
    safety_dims = {"responsible_ai", "responsible ai", "safety"}
    safety_metrics: Dict[str, List[float]] = {}
    for r in results:
        for dim in r.dimension_results:
            if dim.dimension_name.lower().replace("_", " ") in safety_dims:
                for ms in dim.metric_scores:
                    if ms.score is not None:
                        safety_metrics.setdefault(ms.metric_name, []).append(ms.score)
    if not safety_metrics:
        return ["No safety metrics found in evaluation results."]
    lines = ["| Safety Metric | Avg Score | Min | Concerns |",
             "|---------------|-----------|-----|----------|"]
    for name, scores in sorted(safety_metrics.items()):
        avg = statistics.mean(scores)
        concerns = sum(1 for s in scores if s < 0.7)
        concern_str = f"{concerns} ⚠️" if concerns > 0 else "0"
        lines.append(f"| {name} | {avg:.3f} | {min(scores):.3f} | {concern_str} |")
    return lines


def _build_failures_section(results: List[EvaluationResult]) -> List[str]:
    all_failures = []
    for i, r in enumerate(results):
        if r.failures:
            for f in r.failures:
                all_failures.append(f"- Test {i+1}: {f}")
    return all_failures[:20] if all_failures else []


def _build_recommendations(results: List[EvaluationResult]) -> List[str]:
    lines = []
    dim_scores = _collect_dimension_scores(results)
    weak_dims = [(n, statistics.mean(s)) for n, s in dim_scores.items() if statistics.mean(s) < 0.8]
    weak_dims.sort(key=lambda x: x[1])
    if weak_dims:
        lines.append("**Priority areas for improvement:**")
        lines.append("")
        for name, avg in weak_dims[:5]:
            lines.append(f"- **{name}** (avg: {avg:.2f}) — below 0.80 threshold")
    else:
        lines.append("All dimensions are performing above threshold.")

    metric_scores = _collect_metric_scores(results)
    high_var = [(n, statistics.stdev(s)) for n, s in metric_scores.items()
                if len(s) > 2 and statistics.stdev(s) > 0.15]
    high_var.sort(key=lambda x: x[1], reverse=True)
    if high_var:
        lines.append("")
        lines.append("**Inconsistent metrics (high variance):**")
        lines.append("")
        for name, std in high_var[:5]:
            lines.append(f"- **{name}** (std: {std:.3f})")
    return lines


# =============================================================================
# Section Builders — Comparison
# =============================================================================


def _build_delta_table(runs, labels) -> List[str]:
    baseline_metrics = _collect_metric_scores(runs[0])
    lines = ["| Metric | " + " | ".join(labels) + " | Δ (last vs baseline) |",
             "|--------" + "|-------" * len(labels) + "|------|"]

    for metric in sorted(baseline_metrics.keys()):
        row_values = []
        for run in runs:
            ms = _collect_metric_scores(run)
            if metric in ms:
                row_values.append(statistics.mean(ms[metric]))
            else:
                row_values.append(None)

        cells = [f"{v:.3f}" if v is not None else "—" for v in row_values]

        if row_values[0] is not None and row_values[-1] is not None:
            delta = row_values[-1] - row_values[0]
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "—"

        lines.append(f"| {metric} | " + " | ".join(cells) + f" | {delta_str} |")
    return lines


def _build_metric_trend_table(runs, labels) -> List[str]:
    """Flag metrics with >5% change between baseline and last run."""
    baseline_metrics = _collect_metric_scores(runs[0])
    last_metrics = _collect_metric_scores(runs[-1])

    regressions = []
    improvements = []

    for metric in sorted(baseline_metrics.keys()):
        if metric not in last_metrics:
            continue
        base_avg = statistics.mean(baseline_metrics[metric])
        last_avg = statistics.mean(last_metrics[metric])
        delta_pct = ((last_avg - base_avg) / base_avg * 100) if base_avg > 0 else 0

        if delta_pct < -5:
            regressions.append((metric, base_avg, last_avg, delta_pct))
        elif delta_pct > 5:
            improvements.append((metric, base_avg, last_avg, delta_pct))

    lines = []
    if regressions:
        lines.append(f"**{len(regressions)} metric(s) regressed (>5% decline):**")
        lines.append("")
        for name, base, last, pct in sorted(regressions, key=lambda x: x[3]):
            lines.append(f"- ❌ **{name}**: {base:.3f} → {last:.3f} ({pct:+.1f}%)")
        lines.append("")

    if improvements:
        lines.append(f"**{len(improvements)} metric(s) improved (>5% gain):**")
        lines.append("")
        for name, base, last, pct in sorted(improvements, key=lambda x: -x[3]):
            lines.append(f"- ✅ **{name}**: {base:.3f} → {last:.3f} ({pct:+.1f}%)")

    if not regressions and not improvements:
        lines.append("All metrics within ±5% of baseline.")
    return lines


# =============================================================================
# Data Collection Helpers
# =============================================================================


def _collect_dimension_scores(results: List[EvaluationResult]) -> Dict[str, List[float]]:
    dim_scores: Dict[str, List[float]] = {}
    for r in results:
        for dim in r.dimension_results:
            dim_scores.setdefault(dim.dimension_name, []).append(dim.aggregate_score)
    return dim_scores


def _collect_metric_scores(results: List[EvaluationResult]) -> Dict[str, List[float]]:
    metric_scores: Dict[str, List[float]] = {}
    for r in results:
        for dim in r.dimension_results:
            for ms in dim.metric_scores:
                if ms.score is not None:
                    metric_scores.setdefault(ms.metric_name, []).append(ms.score)
    return metric_scores


# =============================================================================
# Plot Generation
# =============================================================================


def _safe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return None


def _generate_ci_plot(outlier_data, images_dir: Path) -> Optional[Path]:
    """Horizontal bar chart with 95% confidence interval error bars."""
    plt = _safe_import_matplotlib()
    if not plt:
        return None
    try:
        names = [d[0] for d in outlier_data]
        means = [d[1] for d in outlier_data]
        ci_lowers = [d[2] for d in outlier_data]
        ci_uppers = [d[3] for d in outlier_data]

        lower_err = [m - l for m, l in zip(means, ci_lowers)]
        upper_err = [u - m for m, u in zip(means, ci_uppers)]

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
        ax.barh(range(len(names)), means, xerr=[lower_err, upper_err],
                align="center", alpha=0.7, color="#2196F3", ecolor="#333", capsize=4)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Score")
        ax.set_title("Metric Scores with 95% Confidence Intervals")
        ax.set_xlim(0, 1.05)
        ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.5, label="Threshold (0.8)")
        ax.legend()
        plt.tight_layout()

        path = images_dir / "confidence_intervals.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"Failed to generate CI plot: {e}")
        return None


def _generate_radar_chart(results: List[EvaluationResult], images_dir: Path) -> Optional[Path]:
    """Radar/spider chart of dimension average scores."""
    plt = _safe_import_matplotlib()
    if not plt:
        return None
    try:
        import numpy as np
        dim_scores = _collect_dimension_scores(results)
        if len(dim_scores) < 3:
            return None

        names = sorted(dim_scores.keys())
        values = [statistics.mean(dim_scores[n]) for n in names]
        values += values[:1]

        angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.fill(angles, values, alpha=0.25, color="#2196F3")
        ax.plot(angles, values, "o-", color="#2196F3", linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("Dimension Scores", pad=20)
        plt.tight_layout()

        path = images_dir / "dimension_radar.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"Failed to generate radar chart: {e}")
        return None


def _generate_histogram(results: List[EvaluationResult], images_dir: Path) -> Optional[Path]:
    """Grid of histograms for metrics with highest variance."""
    plt = _safe_import_matplotlib()
    if not plt:
        return None
    try:
        metric_scores = _collect_metric_scores(results)
        plot_metrics = {k: v for k, v in metric_scores.items() if len(v) >= 5}
        if not plot_metrics:
            return None

        n_metrics = min(len(plot_metrics), 12)
        by_var = sorted(plot_metrics.items(),
                        key=lambda x: statistics.stdev(x[1]) if len(x[1]) > 1 else 0, reverse=True)
        selected = dict(by_var[:n_metrics])

        cols = 3
        rows = (n_metrics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for idx, (name, scores) in enumerate(sorted(selected.items())):
            ax = axes_flat[idx]
            ax.hist(scores, bins=min(10, len(scores)), alpha=0.7, color="#4CAF50", edgecolor="black")
            ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.5)
            ax.set_title(name, fontsize=9)
            ax.set_xlim(0, 1.05)

        for idx in range(n_metrics, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.suptitle("Score Distributions by Metric", fontsize=12)
        plt.tight_layout()

        path = images_dir / "score_histograms.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"Failed to generate histogram: {e}")
        return None


def _generate_correlation_heatmap(results: List[EvaluationResult], images_dir: Path) -> Optional[Path]:
    """Pearson correlation matrix heatmap across metrics."""
    plt = _safe_import_matplotlib()
    if not plt:
        return None
    try:
        import numpy as np
        metric_scores = _collect_metric_scores(results)
        metrics_with_data = {k: v for k, v in metric_scores.items()
                            if len(v) == len(results) and len(v) >= 4}
        if len(metrics_with_data) < 3:
            return None

        names = sorted(metrics_with_data.keys())
        data = np.array([metrics_with_data[n] for n in names])
        corr = np.corrcoef(data)

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), max(6, len(names) * 0.5)))
        im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)

        for i in range(len(names)):
            for j in range(len(names)):
                val = corr[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.7 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

        plt.colorbar(im, ax=ax, label="Correlation")
        ax.set_title("Metric Correlation Heatmap")
        plt.tight_layout()

        path = images_dir / "correlation_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"Failed to generate correlation heatmap: {e}")
        return None


def _generate_side_by_side_chart(runs, labels, images_dir: Path) -> Optional[Path]:
    """Grouped bar chart comparing metric averages across runs."""
    plt = _safe_import_matplotlib()
    if not plt:
        return None
    try:
        import numpy as np

        # Collect averages per metric per run
        all_metrics = set()
        for run in runs:
            all_metrics.update(_collect_metric_scores(run).keys())
        metrics = sorted(all_metrics)

        # Limit to top 12 metrics by baseline score range
        if len(metrics) > 12:
            baseline_ms = _collect_metric_scores(runs[0])
            by_interest = sorted(metrics,
                                 key=lambda m: max(baseline_ms.get(m, [0])) - min(baseline_ms.get(m, [0])),
                                 reverse=True)
            metrics = by_interest[:12]

        n_runs = len(runs)
        x = np.arange(len(metrics))
        width = 0.8 / n_runs
        colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"]

        fig, ax = plt.subplots(figsize=(14, 6))
        for i, (run, label) in enumerate(zip(runs, labels)):
            ms = _collect_metric_scores(run)
            values = [statistics.mean(ms.get(m, [0])) for m in metrics]
            offset = (i - n_runs / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=label, color=colors[i % len(colors)], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title("Side-by-Side Metric Comparison")
        ax.legend()
        ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.3)
        plt.tight_layout()

        path = images_dir / "side_by_side.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"Failed to generate side-by-side chart: {e}")
        return None


def _generate_percentage_change_chart(runs, labels, images_dir: Path) -> Optional[Path]:
    """Bar chart showing percentage change of each metric vs baseline."""
    plt = _safe_import_matplotlib()
    if not plt:
        return None
    try:
        import numpy as np

        baseline_ms = _collect_metric_scores(runs[0])
        last_ms = _collect_metric_scores(runs[-1])

        metrics = sorted(set(baseline_ms.keys()) & set(last_ms.keys()))
        pct_changes = []
        for m in metrics:
            base = statistics.mean(baseline_ms[m])
            last = statistics.mean(last_ms[m])
            pct = ((last - base) / base * 100) if base > 0 else 0
            pct_changes.append(pct)

        # Sort by change magnitude
        sorted_pairs = sorted(zip(metrics, pct_changes), key=lambda x: x[1])
        metrics = [p[0] for p in sorted_pairs]
        pct_changes = [p[1] for p in sorted_pairs]

        fig, ax = plt.subplots(figsize=(10, max(4, len(metrics) * 0.35)))
        colors = ["#F44336" if v < 0 else "#4CAF50" for v in pct_changes]
        ax.barh(range(len(metrics)), pct_changes, color=colors, alpha=0.8)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics, fontsize=8)
        ax.set_xlabel("% Change from Baseline")
        ax.set_title(f"Percentage Change: {labels[0]} → {labels[-1]}")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.axvline(x=-5, color="red", linestyle="--", alpha=0.3, label="-5% threshold")
        ax.axvline(x=5, color="green", linestyle="--", alpha=0.3, label="+5% threshold")
        ax.legend(fontsize=8)
        plt.tight_layout()

        path = images_dir / "percentage_change.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"Failed to generate percentage change chart: {e}")
        return None


def _generate_trend_chart(runs, labels, images_dir: Path) -> Optional[Path]:
    """Line chart of top changing metrics across runs."""
    plt = _safe_import_matplotlib()
    if not plt:
        return None
    try:
        all_metrics = set()
        for run in runs:
            all_metrics.update(_collect_metric_scores(run).keys())

        metric_trends: Dict[str, List[float]] = {}
        for metric in all_metrics:
            avgs = []
            for run in runs:
                ms = _collect_metric_scores(run)
                if metric in ms:
                    avgs.append(statistics.mean(ms[metric]))
                else:
                    avgs.append(None)
            if all(v is not None for v in avgs):
                metric_trends[metric] = avgs

        if not metric_trends:
            return None

        by_change = sorted(metric_trends.items(), key=lambda x: abs(x[1][-1] - x[1][0]), reverse=True)
        selected = dict(by_change[:8])

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(labels))
        for name, values in sorted(selected.items()):
            ax.plot(x, values, "o-", label=name, linewidth=2, markersize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title("Metric Trends Across Runs")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.3)
        plt.tight_layout()

        path = images_dir / "metric_trends.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"Failed to generate trend chart: {e}")
        return None


def _generate_comparison_radar(runs, labels, images_dir: Path) -> Optional[Path]:
    """Overlaid radar chart comparing dimensions across runs."""
    plt = _safe_import_matplotlib()
    if not plt:
        return None
    try:
        import numpy as np

        all_dims = set()
        for run in runs:
            all_dims.update(_collect_dimension_scores(run).keys())

        dims = sorted(all_dims)
        if len(dims) < 3:
            return None

        angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"]

        for i, (run, label) in enumerate(zip(runs, labels)):
            ds = _collect_dimension_scores(run)
            values = [statistics.mean(ds.get(d, [0])) for d in dims]
            values += values[:1]
            color = colors[i % len(colors)]
            ax.plot(angles, values, "o-", color=color, linewidth=2, label=label)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dims, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("Dimension Comparison Across Runs", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()

        path = images_dir / "comparison_radar.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"Failed to generate comparison radar: {e}")
        return None
