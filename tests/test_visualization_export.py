"""Tests for static chart export in VisualizationEngine."""

from uaef.reporting.models import Chart, ChartType
from uaef.reporting.visualization import VisualizationEngine


def _bar_chart() -> Chart:
    return Chart(
        title="Dimension Scores",
        chart_type=ChartType.BAR,
        data={
            "dimensions": ["Tool Calling", "Response Quality"],
            "scores": [0.85, 0.92],
        },
        labels=["Tool Calling", "Response Quality"],
    )


def test_save_chart_svg_writes_svg(tmp_path):
    out = tmp_path / "chart.svg"
    VisualizationEngine().save_chart(_bar_chart(), str(out), format="svg")
    assert "<svg" in out.read_text(encoding="utf-8")


def test_save_chart_png_still_writes_png(tmp_path):
    out = tmp_path / "chart.png"
    VisualizationEngine().save_chart(_bar_chart(), str(out), format="png")
    assert out.read_bytes().startswith(b"\x89PNG")


def test_generate_chart_static_still_returns_png():
    data = VisualizationEngine().generate_chart(_bar_chart(), interactive=False)
    assert data.startswith(b"\x89PNG")
