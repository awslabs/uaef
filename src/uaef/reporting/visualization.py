# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualization engine for generating charts and graphs.

This module provides the VisualizationEngine class for generating interactive
and static visualizations from Chart models. It supports multiple chart types
including bar, line, scatter, heatmap, pie, radar, box, and histogram charts.

The engine uses plotly for interactive HTML visualizations and matplotlib as
a fallback for static image generation (PNG, SVG).

Requirements:
    - 18.1: Generate charts for executive dashboards
    - 18.2: Generate charts for experiment comparison
    - 18.3: Generate charts for dimension deep-dive reports
    - 18.4: Generate charts for safety reports
    - 18.5: Support export to multiple formats (HTML, PNG, SVG)
    - 18.6: Generate visualizations for regression reports

Example:
    >>> from uaef.reporting.models import Chart, ChartType
    >>> from uaef.reporting.visualization import VisualizationEngine
    >>> 
    >>> engine = VisualizationEngine()
    >>> chart = Chart(
    ...     title="Dimension Scores",
    ...     chart_type=ChartType.BAR,
    ...     data={"dimensions": ["Tool", "Response"], "scores": [0.85, 0.92]},
    ...     labels=["Tool", "Response"]
    ... )
    >>> html = engine.generate_chart(chart, interactive=True)
    >>> engine.save_chart(chart, "output.html", format="html")
"""

from typing import Union, Optional, Dict, Any, List
import logging
from pathlib import Path

from uaef.reporting.models import Chart, ChartType

# Optional dependencies - gracefully handle missing libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Engine for generating interactive and static visualizations.
    
    This class provides methods to generate various chart types from Chart models.
    It uses plotly for interactive HTML charts and matplotlib for static images.
    
    Attributes:
        default_theme: Default color theme for charts
        themes: Available color themes
    
    Example:
        >>> engine = VisualizationEngine()
        >>> chart = Chart(
        ...     title="Scores",
        ...     chart_type=ChartType.BAR,
        ...     data={"labels": ["A", "B"], "values": [0.8, 0.9]}
        ... )
        >>> html = engine.generate_chart(chart, interactive=True)
    """
    
    # Default color themes
    THEMES = {
        "default": ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"],
        "professional": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        "pastel": ["#AEC6CF", "#FFB347", "#B39EB5", "#FF6961", "#77DD77", "#FDFD96"],
        "dark": ["#1a1a1a", "#333333", "#4d4d4d", "#666666", "#808080", "#999999"],
        "vibrant": ["#FF1744", "#00E676", "#2979FF", "#FFEA00", "#D500F9", "#00E5FF"],
    }
    
    def __init__(self, default_theme: str = "default"):
        """
        Initialize the visualization engine.
        
        Args:
            default_theme: Name of the default color theme to use
        
        Raises:
            ValueError: If the specified theme doesn't exist
            ImportError: If neither plotly nor matplotlib is available
        """
        if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "Neither plotly nor matplotlib is installed. "
                "Install at least one: pip install plotly or pip install matplotlib"
            )
        
        if default_theme not in self.THEMES:
            raise ValueError(
                f"Unknown theme: {default_theme}. "
                f"Available themes: {list(self.THEMES.keys())}"
            )
        
        self.default_theme = default_theme
        self._current_theme = default_theme
        
        logger.info(
            f"VisualizationEngine initialized with theme '{default_theme}' "
            f"(plotly={'available' if PLOTLY_AVAILABLE else 'unavailable'}, "
            f"matplotlib={'available' if MATPLOTLIB_AVAILABLE else 'unavailable'})"
        )
    
    def apply_theme(self, theme_name: str) -> None:
        """
        Apply a predefined color theme.
        
        Args:
            theme_name: Name of the theme to apply
        
        Raises:
            ValueError: If the theme doesn't exist
        
        Example:
            >>> engine = VisualizationEngine()
            >>> engine.apply_theme("professional")
        """
        if theme_name not in self.THEMES:
            raise ValueError(
                f"Unknown theme: {theme_name}. "
                f"Available themes: {list(self.THEMES.keys())}"
            )
        
        self._current_theme = theme_name
        logger.info(f"Applied theme: {theme_name}")
    
    def _get_colors(self, chart: Chart) -> List[str]:
        """
        Get colors for a chart, using chart colors or theme colors.
        
        Args:
            chart: Chart model
        
        Returns:
            List of color codes
        """
        if chart.colors:
            return chart.colors
        return self.THEMES[self._current_theme]
    
    def generate_chart(
        self,
        chart: Chart,
        interactive: bool = True
    ) -> Union[str, bytes]:
        """
        Generate a chart based on the chart type.
        
        This is the main entry point for chart generation. It dispatches to
        the appropriate chart-specific method based on chart_type.
        
        Args:
            chart: Chart model with data and configuration
            interactive: If True, generate interactive HTML (plotly).
                        If False, generate static image (matplotlib).
        
        Returns:
            For interactive=True: HTML string
            For interactive=False: PNG image bytes
        
        Raises:
            ValueError: If chart type is not supported
            ImportError: If required library is not available
        
        Example:
            >>> chart = Chart(
            ...     title="Test",
            ...     chart_type=ChartType.BAR,
            ...     data={"labels": ["A"], "values": [0.8]}
            ... )
            >>> html = engine.generate_chart(chart, interactive=True)
        """
        logger.info(
            f"Generating {chart.chart_type.value} chart: '{chart.title}' "
            f"(interactive={interactive})"
        )
        
        # Dispatch to appropriate method
        chart_generators = {
            ChartType.BAR: self.generate_bar_chart,
            ChartType.LINE: self.generate_line_chart,
            ChartType.SCATTER: self.generate_scatter_chart,
            ChartType.HEATMAP: self.generate_heatmap,
            ChartType.PIE: self.generate_pie_chart,
            ChartType.RADAR: self.generate_radar_chart,
            ChartType.BOX: self.generate_box_chart,
            ChartType.HISTOGRAM: self.generate_histogram,
        }
        
        generator = chart_generators.get(chart.chart_type)
        if not generator:
            raise ValueError(f"Unsupported chart type: {chart.chart_type}")
        
        return generator(chart, interactive)
    
    def generate_bar_chart(
        self,
        chart: Chart,
        interactive: bool = True
    ) -> Union[str, bytes]:
        """
        Generate a bar chart.
        
        Expected data structure:
            - {"dimensions": [...], "scores": [...]} or
            - {"labels": [...], "values": [...]}
        
        Args:
            chart: Chart model with bar chart data
            interactive: If True, generate interactive HTML
        
        Returns:
            HTML string or PNG bytes
        
        Example:
            >>> chart = Chart(
            ...     title="Scores",
            ...     chart_type=ChartType.BAR,
            ...     data={"labels": ["A", "B"], "values": [0.8, 0.9]}
            ... )
            >>> html = engine.generate_bar_chart(chart, interactive=True)
        """
        # Extract data
        data = chart.data
        labels = data.get("labels") or data.get("dimensions", [])
        values = data.get("values") or data.get("scores", [])
        
        if not labels or not values:
            raise ValueError("Bar chart requires 'labels' and 'values' in data")
        
        colors = self._get_colors(chart)
        
        if interactive and PLOTLY_AVAILABLE:
            return self._generate_bar_chart_plotly(chart, labels, values, colors)
        elif MATPLOTLIB_AVAILABLE:
            return self._generate_bar_chart_matplotlib(chart, labels, values, colors)
        else:
            raise ImportError("No visualization library available")
    
    def _generate_bar_chart_plotly(
        self,
        chart: Chart,
        labels: List[str],
        values: List[float],
        colors: List[str]
    ) -> str:
        """Generate interactive bar chart with plotly."""
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors[:len(labels)],
                text=[f"{v:.2f}" for v in values],
                textposition='auto',
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=chart.title,
            xaxis_title=chart.options.get("xaxis_title", ""),
            yaxis_title=chart.options.get("yaxis_title", "Score"),
            template="plotly_white",
            hovermode="x unified",
            showlegend=False,
        )
        
        # Apply options
        if chart.options.get("horizontal"):
            fig.update_traces(orientation='h')
            fig.update_layout(
                xaxis_title=chart.options.get("xaxis_title", "Score"),
                yaxis_title=chart.options.get("yaxis_title", ""),
            )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def _generate_bar_chart_matplotlib(
        self,
        chart: Chart,
        labels: List[str],
        values: List[float],
        colors: List[str]
    ) -> bytes:
        """Generate static bar chart with matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        bars = ax.bar(
            range(len(labels)),
            values,
            color=colors[:len(labels)]
        )
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom'
            )
        
        # Set labels and title
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(chart.options.get("yaxis_title", "Score"))
        ax.set_title(chart.title)
        ax.grid(axis='y', alpha=0.3)
        
        # Save to bytes
        from io import BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    
    def generate_line_chart(
        self,
        chart: Chart,
        interactive: bool = True
    ) -> Union[str, bytes]:
        """
        Generate a line chart.
        
        Expected data structure:
            - {"timestamps": [...], "values": [...]} or
            - {"x": [...], "y": [...]} or
            - {"labels": [...], "values": [...]}
        
        Args:
            chart: Chart model with line chart data
            interactive: If True, generate interactive HTML
        
        Returns:
            HTML string or PNG bytes
        """
        data = chart.data
        x_values = data.get("x") or data.get("timestamps") or data.get("labels", [])
        y_values = data.get("y") or data.get("values", [])
        
        if not x_values or not y_values:
            raise ValueError("Line chart requires x and y values in data")
        
        colors = self._get_colors(chart)
        
        if interactive and PLOTLY_AVAILABLE:
            return self._generate_line_chart_plotly(chart, x_values, y_values, colors)
        elif MATPLOTLIB_AVAILABLE:
            return self._generate_line_chart_matplotlib(chart, x_values, y_values, colors)
        else:
            raise ImportError("No visualization library available")
    
    def _generate_line_chart_plotly(
        self,
        chart: Chart,
        x_values: List,
        y_values: List[float],
        colors: List[str]
    ) -> str:
        """Generate interactive line chart with plotly."""
        fig = go.Figure()
        
        # Add line trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            line=dict(color=colors[0], width=2),
            marker=dict(size=8),
            name=chart.title
        ))
        
        # Add trend line if requested
        if chart.options.get("show_trend_line"):
            # Simple linear trend
            if MATPLOTLIB_AVAILABLE:
                z = np.polyfit(range(len(y_values)), y_values, 1)
                p = np.poly1d(z)
                trend_y = [p(i) for i in range(len(y_values))]
                
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=trend_y,
                    mode='lines',
                    line=dict(color=colors[1] if len(colors) > 1 else colors[0], dash='dash'),
                    name='Trend'
                ))
        
        # Update layout
        fig.update_layout(
            title=chart.title,
            xaxis_title=chart.options.get("xaxis_title", ""),
            yaxis_title=chart.options.get("yaxis_title", "Value"),
            template="plotly_white",
            hovermode="x unified",
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
    
    def _generate_line_chart_matplotlib(
        self,
        chart: Chart,
        x_values: List,
        y_values: List[float],
        colors: List[str]
    ) -> bytes:
        """Generate static line chart with matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot line
        ax.plot(x_values, y_values, marker='o', color=colors[0], linewidth=2, markersize=8)
        
        # Add trend line if requested
        if chart.options.get("show_trend_line"):
            z = np.polyfit(range(len(y_values)), y_values, 1)
            p = np.poly1d(z)
            trend_y = [p(i) for i in range(len(y_values))]
            ax.plot(x_values, trend_y, '--', color=colors[1] if len(colors) > 1 else colors[0], alpha=0.7, label='Trend')
            ax.legend()
        
        # Set labels and title
        ax.set_xlabel(chart.options.get("xaxis_title", ""))
        ax.set_ylabel(chart.options.get("yaxis_title", "Value"))
        ax.set_title(chart.title)
        ax.grid(alpha=0.3)
        
        # Rotate x-axis labels if they're strings
        if x_values and isinstance(x_values[0], str):
            plt.xticks(rotation=45, ha='right')
        
        # Save to bytes
        from io import BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    
    def generate_scatter_chart(
        self,
        chart: Chart,
        interactive: bool = True
    ) -> Union[str, bytes]:
        """
        Generate a scatter chart.
        
        Expected data structure:
            - {"x": [...], "y": [...]}
        
        Args:
            chart: Chart model with scatter chart data
            interactive: If True, generate interactive HTML
        
        Returns:
            HTML string or PNG bytes
        """
        data = chart.data
        x_values = data.get("x", [])
        y_values = data.get("y", [])
        
        if not x_values or not y_values:
            raise ValueError("Scatter chart requires 'x' and 'y' in data")
        
        colors = self._get_colors(chart)
        
        if interactive and PLOTLY_AVAILABLE:
            return self._generate_scatter_chart_plotly(chart, x_values, y_values, colors)
        elif MATPLOTLIB_AVAILABLE:
            return self._generate_scatter_chart_matplotlib(chart, x_values, y_values, colors)
        else:
            raise ImportError("No visualization library available")
    
    def _generate_scatter_chart_plotly(
        self,
        chart: Chart,
        x_values: List[float],
        y_values: List[float],
        colors: List[str]
    ) -> str:
        """Generate interactive scatter chart with plotly."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            marker=dict(
                size=10,
                color=colors[0],
                line=dict(width=1, color='white')
            ),
            name=chart.title
        ))
        
        # Update layout
        fig.update_layout(
            title=chart.title,
            xaxis_title=chart.options.get("xaxis_title", "X"),
            yaxis_title=chart.options.get("yaxis_title", "Y"),
            template="plotly_white",
            hovermode="closest",
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
    
    def _generate_scatter_chart_matplotlib(
        self,
        chart: Chart,
        x_values: List[float],
        y_values: List[float],
        colors: List[str]
    ) -> bytes:
        """Generate static scatter chart with matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(x_values, y_values, c=colors[0], s=100, alpha=0.6, edgecolors='white', linewidth=1)
        
        ax.set_xlabel(chart.options.get("xaxis_title", "X"))
        ax.set_ylabel(chart.options.get("yaxis_title", "Y"))
        ax.set_title(chart.title)
        ax.grid(alpha=0.3)
        
        # Save to bytes
        from io import BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    
    def generate_heatmap(
        self,
        chart: Chart,
        interactive: bool = True
    ) -> Union[str, bytes]:
        """
        Generate a heatmap.
        
        Expected data structure:
            - {"matrix": [[...]], "x_labels": [...], "y_labels": [...]}
        
        Args:
            chart: Chart model with heatmap data
            interactive: If True, generate interactive HTML
        
        Returns:
            HTML string or PNG bytes
        """
        data = chart.data
        matrix = data.get("matrix", [])
        x_labels = data.get("x_labels", [])
        y_labels = data.get("y_labels", [])
        
        if not matrix:
            raise ValueError("Heatmap requires 'matrix' in data")
        
        colors = self._get_colors(chart)
        
        if interactive and PLOTLY_AVAILABLE:
            return self._generate_heatmap_plotly(chart, matrix, x_labels, y_labels, colors)
        elif MATPLOTLIB_AVAILABLE:
            return self._generate_heatmap_matplotlib(chart, matrix, x_labels, y_labels, colors)
        else:
            raise ImportError("No visualization library available")
    
    def _generate_heatmap_plotly(
        self,
        chart: Chart,
        matrix: List[List[float]],
        x_labels: List[str],
        y_labels: List[str],
        colors: List[str]
    ) -> str:
        """Generate interactive heatmap with plotly."""
        # Use a color scale
        colorscale = chart.options.get("colorscale", "Viridis")
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=x_labels if x_labels else None,
            y=y_labels if y_labels else None,
            colorscale=colorscale,
            text=matrix,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=chart.title,
            xaxis_title=chart.options.get("xaxis_title", ""),
            yaxis_title=chart.options.get("yaxis_title", ""),
            template="plotly_white",
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
    
    def _generate_heatmap_matplotlib(
        self,
        chart: Chart,
        matrix: List[List[float]],
        x_labels: List[str],
        y_labels: List[str],
        colors: List[str]
    ) -> bytes:
        """Generate static heatmap with matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add text annotations
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                text = ax.text(j, i, f'{matrix[i][j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8)
        
        ax.set_title(chart.title)
        
        # Save to bytes
        from io import BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def generate_pie_chart(
        self,
        chart: Chart,
        interactive: bool = True
    ) -> Union[str, bytes]:
        """
        Generate a pie chart.
        
        Expected data structure:
            - {"labels": [...], "values": [...]}
        
        Args:
            chart: Chart model with pie chart data
            interactive: If True, generate interactive HTML
        
        Returns:
            HTML string or PNG bytes
        """
        data = chart.data
        labels = data.get("labels", [])
        values = data.get("values", [])
        
        if not labels or not values:
            raise ValueError("Pie chart requires 'labels' and 'values' in data")
        
        colors = self._get_colors(chart)
        
        if interactive and PLOTLY_AVAILABLE:
            return self._generate_pie_chart_plotly(chart, labels, values, colors)
        elif MATPLOTLIB_AVAILABLE:
            return self._generate_pie_chart_matplotlib(chart, labels, values, colors)
        else:
            raise ImportError("No visualization library available")
    
    def _generate_pie_chart_plotly(
        self,
        chart: Chart,
        labels: List[str],
        values: List[float],
        colors: List[str]
    ) -> str:
        """Generate interactive pie chart with plotly."""
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors[:len(labels)]),
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        )])
        
        fig.update_layout(
            title=chart.title,
            template="plotly_white",
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
    
    def _generate_pie_chart_matplotlib(
        self,
        chart: Chart,
        labels: List[str],
        values: List[float],
        colors: List[str]
    ) -> bytes:
        """Generate static pie chart with matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.pie(
            values,
            labels=labels,
            colors=colors[:len(labels)],
            autopct='%1.1f%%',
            startangle=90
        )
        
        ax.set_title(chart.title)
        
        # Save to bytes
        from io import BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    
    def generate_radar_chart(
        self,
        chart: Chart,
        interactive: bool = True
    ) -> Union[str, bytes]:
        """
        Generate a radar (spider) chart.
        
        Expected data structure:
            - {"dimensions": [...], "scores": [...]}
        
        Args:
            chart: Chart model with radar chart data
            interactive: If True, generate interactive HTML
        
        Returns:
            HTML string or PNG bytes
        """
        data = chart.data
        dimensions = data.get("dimensions", [])
        scores = data.get("scores", [])
        
        if not dimensions or not scores:
            raise ValueError("Radar chart requires 'dimensions' and 'scores' in data")
        
        colors = self._get_colors(chart)
        
        if interactive and PLOTLY_AVAILABLE:
            return self._generate_radar_chart_plotly(chart, dimensions, scores, colors)
        elif MATPLOTLIB_AVAILABLE:
            return self._generate_radar_chart_matplotlib(chart, dimensions, scores, colors)
        else:
            raise ImportError("No visualization library available")
    
    def _generate_radar_chart_plotly(
        self,
        chart: Chart,
        dimensions: List[str],
        scores: List[float],
        colors: List[str]
    ) -> str:
        """Generate interactive radar chart with plotly."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=dimensions,
            fill='toself',
            fillcolor=colors[0],
            line=dict(color=colors[0]),
            opacity=0.6,
            name=chart.title
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=chart.title,
            template="plotly_white",
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
    
    def _generate_radar_chart_matplotlib(
        self,
        chart: Chart,
        dimensions: List[str],
        scores: List[float],
        colors: List[str]
    ) -> bytes:
        """Generate static radar chart with matplotlib."""
        # Number of variables
        num_vars = len(dimensions)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Complete the circle
        scores_plot = scores + [scores[0]]
        angles_plot = angles + [angles[0]]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles_plot, scores_plot, 'o-', linewidth=2, color=colors[0])
        ax.fill(angles_plot, scores_plot, alpha=0.25, color=colors[0])
        
        # Fix axis to go in the right order
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Set labels
        ax.set_xticks(angles)
        ax.set_xticklabels(dimensions)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        ax.set_title(chart.title, pad=20)
        ax.grid(True)
        
        # Save to bytes
        from io import BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    
    def generate_box_chart(
        self,
        chart: Chart,
        interactive: bool = True
    ) -> Union[str, bytes]:
        """
        Generate a box plot.
        
        Expected data structure:
            - {"data": [...], "labels": [...]} or
            - {"data": [[...], [...]], "labels": [...]}
        
        Args:
            chart: Chart model with box plot data
            interactive: If True, generate interactive HTML
        
        Returns:
            HTML string or PNG bytes
        """
        data = chart.data
        box_data = data.get("data", [])
        labels = data.get("labels", [])
        
        if not box_data:
            raise ValueError("Box chart requires 'data' in data")
        
        colors = self._get_colors(chart)
        
        if interactive and PLOTLY_AVAILABLE:
            return self._generate_box_chart_plotly(chart, box_data, labels, colors)
        elif MATPLOTLIB_AVAILABLE:
            return self._generate_box_chart_matplotlib(chart, box_data, labels, colors)
        else:
            raise ImportError("No visualization library available")
    
    def _generate_box_chart_plotly(
        self,
        chart: Chart,
        box_data: Union[List[float], List[List[float]]],
        labels: List[str],
        colors: List[str]
    ) -> str:
        """Generate interactive box plot with plotly."""
        fig = go.Figure()
        
        # Handle single or multiple box plots
        if box_data and isinstance(box_data[0], list):
            # Multiple box plots
            for i, (data_series, label) in enumerate(zip(box_data, labels)):
                fig.add_trace(go.Box(
                    y=data_series,
                    name=label,
                    marker_color=colors[i % len(colors)]
                ))
        else:
            # Single box plot
            fig.add_trace(go.Box(
                y=box_data,
                name=labels[0] if labels else chart.title,
                marker_color=colors[0]
            ))
        
        fig.update_layout(
            title=chart.title,
            yaxis_title=chart.options.get("yaxis_title", "Value"),
            template="plotly_white",
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
    
    def _generate_box_chart_matplotlib(
        self,
        chart: Chart,
        box_data: Union[List[float], List[List[float]]],
        labels: List[str],
        colors: List[str]
    ) -> bytes:
        """Generate static box plot with matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Handle single or multiple box plots
        if box_data and isinstance(box_data[0], list):
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            # Color each box
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
        else:
            bp = ax.boxplot([box_data], labels=labels if labels else [chart.title], patch_artist=True)
            bp['boxes'][0].set_facecolor(colors[0])
        
        ax.set_ylabel(chart.options.get("yaxis_title", "Value"))
        ax.set_title(chart.title)
        ax.grid(axis='y', alpha=0.3)
        
        # Save to bytes
        from io import BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    
    def generate_histogram(
        self,
        chart: Chart,
        interactive: bool = True
    ) -> Union[str, bytes]:
        """
        Generate a histogram.
        
        Expected data structure:
            - {"values": [...], "bins": int}
        
        Args:
            chart: Chart model with histogram data
            interactive: If True, generate interactive HTML
        
        Returns:
            HTML string or PNG bytes
        """
        data = chart.data
        values = data.get("values", [])
        bins = data.get("bins", 10)
        
        if not values:
            raise ValueError("Histogram requires 'values' in data")
        
        colors = self._get_colors(chart)
        
        if interactive and PLOTLY_AVAILABLE:
            return self._generate_histogram_plotly(chart, values, bins, colors)
        elif MATPLOTLIB_AVAILABLE:
            return self._generate_histogram_matplotlib(chart, values, bins, colors)
        else:
            raise ImportError("No visualization library available")
    
    def _generate_histogram_plotly(
        self,
        chart: Chart,
        values: List[float],
        bins: int,
        colors: List[str]
    ) -> str:
        """Generate interactive histogram with plotly."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=bins,
            marker_color=colors[0],
            name=chart.title
        ))
        
        fig.update_layout(
            title=chart.title,
            xaxis_title=chart.options.get("xaxis_title", "Value"),
            yaxis_title=chart.options.get("yaxis_title", "Frequency"),
            template="plotly_white",
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
    
    def _generate_histogram_matplotlib(
        self,
        chart: Chart,
        values: List[float],
        bins: int,
        colors: List[str]
    ) -> bytes:
        """Generate static histogram with matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(values, bins=bins, color=colors[0], alpha=0.7, edgecolor='black')
        
        ax.set_xlabel(chart.options.get("xaxis_title", "Value"))
        ax.set_ylabel(chart.options.get("yaxis_title", "Frequency"))
        ax.set_title(chart.title)
        ax.grid(axis='y', alpha=0.3)
        
        # Save to bytes
        from io import BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    
    def save_chart(
        self,
        chart: Chart,
        output_path: str,
        format: str = "html",
        interactive: bool = True
    ) -> None:
        """
        Generate and save a chart to a file.
        
        Args:
            chart: Chart model to generate
            output_path: Path where to save the chart
            format: Output format ("html", "png", "svg")
            interactive: If True, generate interactive chart (only for HTML)
        
        Raises:
            ValueError: If format is not supported
            IOError: If file cannot be written
        
        Example:
            >>> chart = Chart(
            ...     title="Test",
            ...     chart_type=ChartType.BAR,
            ...     data={"labels": ["A"], "values": [0.8]}
            ... )
            >>> engine.save_chart(chart, "output.html", format="html")
        """
        format = format.lower()
        
        if format == "html":
            if not PLOTLY_AVAILABLE:
                raise ImportError("plotly is required for HTML output")
            
            html_content = self.generate_chart(chart, interactive=True)
            
            # Wrap in full HTML if needed
            if not html_content.startswith("<!DOCTYPE") and not html_content.startswith("<html"):
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{chart.title}</title>
</head>
<body>
    {html_content}
</body>
</html>"""
            
            Path(output_path).write_text(html_content, encoding='utf-8')
            logger.info(f"Saved HTML chart to {output_path}")
        
        elif format in ["png", "svg"]:
            if not MATPLOTLIB_AVAILABLE:
                raise ImportError("matplotlib is required for PNG/SVG output")
            
            # Generate with matplotlib
            image_bytes = self.generate_chart(chart, interactive=False)
            
            if format == "svg":
                # Re-generate as SVG
                # This is a simplified approach - in production, you'd want to
                # modify the matplotlib generation to support SVG directly
                raise NotImplementedError("SVG format not yet implemented")
            
            Path(output_path).write_bytes(image_bytes)
            logger.info(f"Saved {format.upper()} chart to {output_path}")
        
        else:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported formats: html, png, svg"
            )
