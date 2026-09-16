# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dashboard and reporting data models for visualization."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class ChartType(str, Enum):
    """Type of chart for visualization."""
    
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    PIE = "pie"
    RADAR = "radar"
    BOX = "box"
    HISTOGRAM = "histogram"


class Chart(BaseModel):
    """
    Chart configuration for dashboard visualization.
    
    Attributes:
        chart_id: Unique identifier for the chart
        title: Chart title
        chart_type: Type of chart (bar, line, scatter, heatmap, etc.)
        data: Chart data (structure depends on chart_type)
        labels: Labels for data points (x-axis labels, categories, etc.)
        colors: Color scheme for the chart
        options: Additional chart configuration options
        description: Optional description of what the chart shows
    
    Examples:
        >>> # Bar chart for dimension scores
        >>> chart = Chart(
        ...     title="Dimension Scores",
        ...     chart_type=ChartType.BAR,
        ...     data={
        ...         "dimensions": ["Tool Calling", "Response Quality", "Safety"],
        ...         "scores": [0.85, 0.92, 0.88]
        ...     },
        ...     labels=["Tool Calling", "Response Quality", "Safety"],
        ...     colors=["#4CAF50", "#2196F3", "#FF9800"]
        ... )
        
        >>> # Line chart for metric trends
        >>> chart = Chart(
        ...     title="Tool Accuracy Over Time",
        ...     chart_type=ChartType.LINE,
        ...     data={
        ...         "timestamps": ["2024-01-01", "2024-01-02", "2024-01-03"],
        ...         "values": [0.80, 0.85, 0.88]
        ...     },
        ...     labels=["Jan 1", "Jan 2", "Jan 3"],
        ...     options={"show_trend_line": True}
        ... )
        
        >>> # Heatmap for confusion matrix
        >>> chart = Chart(
        ...     title="Tool Selection Confusion Matrix",
        ...     chart_type=ChartType.HEATMAP,
        ...     data={
        ...         "matrix": [[10, 2, 1], [1, 15, 0], [0, 1, 12]],
        ...         "x_labels": ["search", "calculator", "database"],
        ...         "y_labels": ["search", "calculator", "database"]
        ...     },
        ...     colors=["#FFFFFF", "#4CAF50"]
        ... )
    """
    
    chart_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the chart"
    )
    title: str = Field(
        ...,
        description="Chart title",
        min_length=1
    )
    chart_type: ChartType = Field(
        ...,
        description="Type of chart (bar, line, scatter, heatmap, etc.)"
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Chart data (structure depends on chart_type)"
    )
    labels: List[str] = Field(
        default_factory=list,
        description="Labels for data points (x-axis labels, categories, etc.)"
    )
    colors: List[str] = Field(
        default_factory=list,
        description="Color scheme for the chart"
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional chart configuration options"
    )
    description: Optional[str] = Field(
        None,
        description="Optional description of what the chart shows"
    )
    
    @field_validator("chart_id")
    @classmethod
    def validate_chart_id(cls, v: UUID) -> UUID:
        """Validate that chart_id is not None."""
        if v is None:
            raise ValueError("chart_id cannot be None")
        return v
    
    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate that title is not empty."""
        if not v or not v.strip():
            raise ValueError("Chart title cannot be empty")
        return v.strip()
    
    @field_validator("data")
    @classmethod
    def validate_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that data is not empty."""
        if not v:
            raise ValueError("Chart data cannot be empty")
        return v
    
    @field_validator("colors")
    @classmethod
    def validate_colors(cls, v: List[str]) -> List[str]:
        """Validate that colors are valid hex codes or color names."""
        for color in v:
            if not color:
                raise ValueError("Color cannot be empty string")
            # Basic validation for hex colors (starts with #)
            if color.startswith("#"):
                hex_part = color[1:]
                if len(hex_part) not in [3, 6, 8]:  # RGB, RRGGBB, RRGGBBAA
                    raise ValueError(f"Invalid hex color format: {color}")
                # Check if all characters are valid hex digits
                if not all(c in "0123456789ABCDEFabcdef" for c in hex_part):
                    raise ValueError(f"Invalid hex color: {color} contains non-hex characters")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "chart_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Dimension Scores",
                "chart_type": "bar",
                "data": {
                    "dimensions": ["Tool Calling", "Response Quality", "Safety"],
                    "scores": [0.85, 0.92, 0.88]
                },
                "labels": ["Tool Calling", "Response Quality", "Safety"],
                "colors": ["#4CAF50", "#2196F3", "#FF9800"],
                "options": {
                    "show_values": True,
                    "horizontal": False
                },
                "description": "Overall dimension scores for the experiment run"
            }
        }


class Dashboard(BaseModel):
    """
    Dashboard containing multiple charts and visualizations.
    
    Attributes:
        dashboard_id: Unique identifier for the dashboard
        title: Dashboard title
        description: Dashboard description
        charts: List of charts in the dashboard
        metadata: Additional dashboard metadata (experiment_id, run_id, etc.)
        created_at: When the dashboard was created
        layout: Layout configuration for chart positioning
    
    Examples:
        >>> # Executive dashboard
        >>> dashboard = Dashboard(
        ...     title="Executive Dashboard - Customer Support Agent",
        ...     description="High-level metrics for experiment run baseline-v1",
        ...     charts=[
        ...         Chart(title="Overall Score", chart_type=ChartType.BAR, data={"score": [0.87]}),
        ...         Chart(title="Dimension Breakdown", chart_type=ChartType.RADAR, data={...})
        ...     ],
        ...     metadata={
        ...         "experiment_id": "550e8400-e29b-41d4-a716-446655440000",
        ...         "run_id": "660e8400-e29b-41d4-a716-446655440001",
        ...         "dashboard_type": "executive"
        ...     }
        ... )
        
        >>> # Comparison dashboard
        >>> dashboard = Dashboard(
        ...     title="Experiment Comparison: v1 vs v2",
        ...     description="Side-by-side comparison of baseline and new version",
        ...     charts=[
        ...         Chart(title="Metric Deltas", chart_type=ChartType.BAR, data={...}),
        ...         Chart(title="Trend Over Time", chart_type=ChartType.LINE, data={...})
        ...     ],
        ...     metadata={
        ...         "baseline_run_id": "660e8400-e29b-41d4-a716-446655440001",
        ...         "comparison_run_id": "770e8400-e29b-41d4-a716-446655440002",
        ...         "dashboard_type": "comparison"
        ...     }
        ... )
    """
    
    dashboard_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the dashboard"
    )
    title: str = Field(
        ...,
        description="Dashboard title",
        min_length=1
    )
    description: Optional[str] = Field(
        None,
        description="Dashboard description"
    )
    charts: List[Chart] = Field(
        ...,
        description="List of charts in the dashboard",
        min_length=1
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional dashboard metadata (experiment_id, run_id, etc.)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the dashboard was created"
    )
    layout: Dict[str, Any] = Field(
        default_factory=dict,
        description="Layout configuration for chart positioning"
    )
    
    @field_validator("dashboard_id")
    @classmethod
    def validate_dashboard_id(cls, v: UUID) -> UUID:
        """Validate that dashboard_id is not None."""
        if v is None:
            raise ValueError("dashboard_id cannot be None")
        return v
    
    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate that title is not empty."""
        if not v or not v.strip():
            raise ValueError("Dashboard title cannot be empty")
        return v.strip()
    
    @field_validator("charts")
    @classmethod
    def validate_charts(cls, v: List[Chart]) -> List[Chart]:
        """Validate that charts list is not empty."""
        if not v:
            raise ValueError("Dashboard must contain at least one chart")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "dashboard_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Executive Dashboard - Customer Support Agent",
                "description": "High-level metrics for experiment run baseline-v1",
                "charts": [
                    {
                        "chart_id": "660e8400-e29b-41d4-a716-446655440001",
                        "title": "Overall Score",
                        "chart_type": "bar",
                        "data": {"score": [0.87]},
                        "labels": ["Overall"],
                        "colors": ["#4CAF50"]
                    }
                ],
                "metadata": {
                    "experiment_id": "770e8400-e29b-41d4-a716-446655440002",
                    "run_id": "880e8400-e29b-41d4-a716-446655440003",
                    "dashboard_type": "executive"
                },
                "created_at": "2024-01-15T10:00:00Z",
                "layout": {
                    "grid_columns": 2,
                    "chart_positions": {
                        "660e8400-e29b-41d4-a716-446655440001": {"row": 0, "col": 0}
                    }
                }
            }
        }


class ReportFormat(str, Enum):
    """Format for report export."""
    
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    MARKDOWN = "markdown"


class ReportType(str, Enum):
    """Type of report."""
    
    EXECUTIVE = "executive"
    COMPARISON = "comparison"
    DIMENSION_DEEP_DIVE = "dimension_deep_dive"
    SAFETY = "safety"
    REGRESSION = "regression"
    CUSTOM = "custom"


class Report(BaseModel):
    """
    Generated report with dashboard and exportable content.
    
    Attributes:
        report_id: Unique identifier for the report
        title: Report title
        report_type: Type of report (executive, comparison, safety, etc.)
        dashboard: Dashboard with visualizations
        summary: Executive summary text
        sections: Report sections with detailed content
        metadata: Additional report metadata
        created_at: When the report was created
        export_formats: Available export formats for this report
    
    Examples:
        >>> # Executive report
        >>> report = Report(
        ...     title="Executive Report - Customer Support Agent v1",
        ...     report_type=ReportType.EXECUTIVE,
        ...     dashboard=Dashboard(...),
        ...     summary="Overall performance: 87%. All dimensions above threshold.",
        ...     sections=[
        ...         {
        ...             "title": "Key Findings",
        ...             "content": "Tool calling accuracy improved by 5%..."
        ...         },
        ...         {
        ...             "title": "Recommendations",
        ...             "content": "Consider optimizing response latency..."
        ...         }
        ...     ],
        ...     metadata={
        ...         "experiment_id": "550e8400-e29b-41d4-a716-446655440000",
        ...         "run_id": "660e8400-e29b-41d4-a716-446655440001"
        ...     },
        ...     export_formats=[ReportFormat.HTML, ReportFormat.PDF, ReportFormat.JSON]
        ... )
        
        >>> # Safety report
        >>> report = Report(
        ...     title="Safety Report - Responsible AI Assessment",
        ...     report_type=ReportType.SAFETY,
        ...     dashboard=Dashboard(...),
        ...     summary="2 safety concerns detected requiring attention.",
        ...     sections=[
        ...         {
        ...             "title": "Safety Violations",
        ...             "content": "Detected potential bias in 2 responses...",
        ...             "severity": "medium"
        ...         }
        ...     ],
        ...     metadata={"focus": "responsible_ai"},
        ...     export_formats=[ReportFormat.HTML, ReportFormat.PDF]
        ... )
        
        >>> # Regression report
        >>> report = Report(
        ...     title="Regression Report - v1 vs v2",
        ...     report_type=ReportType.REGRESSION,
        ...     dashboard=Dashboard(...),
        ...     summary="3 regressions detected in tool calling dimension.",
        ...     sections=[
        ...         {
        ...             "title": "Regressions",
        ...             "content": "Tool accuracy decreased by 8%...",
        ...             "regressions": [
        ...                 {"metric": "tool_accuracy", "delta": -0.08, "severity": "high"}
        ...             ]
        ...         }
        ...     ],
        ...     metadata={
        ...         "baseline_run_id": "660e8400-e29b-41d4-a716-446655440001",
        ...         "comparison_run_id": "770e8400-e29b-41d4-a716-446655440002"
        ...     }
        ... )
    """
    
    report_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the report"
    )
    title: str = Field(
        ...,
        description="Report title",
        min_length=1
    )
    report_type: ReportType = Field(
        ...,
        description="Type of report (executive, comparison, safety, etc.)"
    )
    dashboard: Dashboard = Field(
        ...,
        description="Dashboard with visualizations"
    )
    summary: str = Field(
        ...,
        description="Executive summary text",
        min_length=1
    )
    sections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Report sections with detailed content"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional report metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the report was created"
    )
    export_formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.JSON, ReportFormat.HTML],
        description="Available export formats for this report"
    )
    
    @field_validator("report_id")
    @classmethod
    def validate_report_id(cls, v: UUID) -> UUID:
        """Validate that report_id is not None."""
        if v is None:
            raise ValueError("report_id cannot be None")
        return v
    
    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate that title is not empty."""
        if not v or not v.strip():
            raise ValueError("Report title cannot be empty")
        return v.strip()
    
    @field_validator("summary")
    @classmethod
    def validate_summary(cls, v: str) -> str:
        """Validate that summary is not empty."""
        if not v or not v.strip():
            raise ValueError("Report summary cannot be empty")
        return v.strip()
    
    @field_validator("sections")
    @classmethod
    def validate_sections(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate that each section has required fields."""
        for i, section in enumerate(v):
            if "title" not in section:
                raise ValueError(f"Section {i} missing required field 'title'")
            if "content" not in section:
                raise ValueError(f"Section {i} missing required field 'content'")
            if not section["title"] or not section["title"].strip():
                raise ValueError(f"Section {i} title cannot be empty")
            if not section["content"] or not section["content"].strip():
                raise ValueError(f"Section {i} content cannot be empty")
        return v
    
    @field_validator("export_formats")
    @classmethod
    def validate_export_formats(cls, v: List[ReportFormat]) -> List[ReportFormat]:
        """Validate that at least one export format is specified."""
        if not v:
            raise ValueError("Report must support at least one export format")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "report_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Executive Report - Customer Support Agent v1",
                "report_type": "executive",
                "dashboard": {
                    "dashboard_id": "660e8400-e29b-41d4-a716-446655440001",
                    "title": "Executive Dashboard",
                    "charts": []
                },
                "summary": "Overall performance: 87%. All dimensions above threshold.",
                "sections": [
                    {
                        "title": "Key Findings",
                        "content": "Tool calling accuracy improved by 5% compared to baseline."
                    },
                    {
                        "title": "Recommendations",
                        "content": "Consider optimizing response latency for better user experience."
                    }
                ],
                "metadata": {
                    "experiment_id": "770e8400-e29b-41d4-a716-446655440002",
                    "run_id": "880e8400-e29b-41d4-a716-446655440003",
                    "evaluation_count": 100
                },
                "created_at": "2024-01-15T10:00:00Z",
                "export_formats": ["json", "html", "pdf"]
            }
        }
