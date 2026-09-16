# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report exporter for multiple output formats."""

import csv
import json
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional, Union

try:
    from jinja2 import Environment, PackageLoader, select_autoescape
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from .models import Dashboard, Report, ReportFormat


class ReportExporter:
    """
    Export reports and dashboards to various formats.
    
    Supports JSON, HTML, PDF, and CSV export formats with professional styling
    and templates.
    
    Examples:
        >>> exporter = ReportExporter()
        >>> 
        >>> # Export to JSON
        >>> json_str = exporter.export_to_json(report)
        >>> 
        >>> # Export to HTML file
        >>> exporter.export_to_html(report, output_path="report.html")
        >>> 
        >>> # Export to PDF
        >>> pdf_bytes = exporter.export_to_pdf(report)
        >>> 
        >>> # Export tabular data to CSV
        >>> csv_str = exporter.export_to_csv(dashboard)
    """
    
    def __init__(self):
        """Initialize the report exporter with Jinja2 templates."""
        # Set up Jinja2 environment for HTML templates
        self.jinja_env = None
        if JINJA2_AVAILABLE:
            try:
                self.jinja_env = Environment(  # nosemgrep: python.flask.security.xss.audit.direct-use-of-jinja2.direct-use-of-jinja2 -- False positive: HTML escaping is enabled via autoescape=select_autoescape(["html","xml"]); templates are packaged (PackageLoader), not user-supplied.
                    loader=PackageLoader("uaef.reporting", "templates"),
                    autoescape=select_autoescape(["html", "xml"])
                )
            except Exception:
                # If templates directory doesn't exist, jinja_env stays None
                pass
    
    def export_to_json(
        self,
        report: Union[Report, Dashboard],
        output_path: Optional[Union[str, Path]] = None,
        indent: int = 2
    ) -> str:
        """
        Export report or dashboard to JSON format.
        
        Uses Pydantic's model_dump() method to serialize the data with proper
        type handling for UUIDs, datetimes, and enums.
        
        Args:
            report: Report or Dashboard to export
            output_path: Optional file path to write JSON to
            indent: Number of spaces for JSON indentation (default: 2)
        
        Returns:
            JSON string representation
        
        Raises:
            ValueError: If report is None or invalid
            IOError: If output_path is specified but file cannot be written
        
        Examples:
            >>> exporter = ReportExporter()
            >>> json_str = exporter.export_to_json(report)
            >>> print(json_str[:100])
            {
              "report_id": "550e8400-e29b-41d4-a716-446655440000",
              "title": "Executive Report",
            ...
            
            >>> # Write to file
            >>> exporter.export_to_json(report, output_path="report.json")
        """
        if report is None:
            raise ValueError("Report cannot be None")
        
        try:
            # Use Pydantic's model_dump for proper serialization
            data = report.model_dump(mode="json")
            json_str = json.dumps(data, indent=indent, ensure_ascii=False)
            
            # Write to file if path specified
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json_str, encoding="utf-8")
            
            return json_str
            
        except Exception as e:
            raise ValueError(f"Failed to export to JSON: {str(e)}") from e
    
    def export_to_html(
        self,
        report: Union[Report, Dashboard],
        output_path: Optional[Union[str, Path]] = None,
        template_name: Optional[str] = None
    ) -> str:
        """
        Export report or dashboard to HTML format with professional styling.
        
        Uses Jinja2 templates for rendering. If templates are not available,
        generates basic HTML structure.
        
        Args:
            report: Report or Dashboard to export
            output_path: Optional file path to write HTML to
            template_name: Optional custom template name (default: auto-detect)
        
        Returns:
            HTML string
        
        Raises:
            ValueError: If report is None or invalid
            IOError: If output_path is specified but file cannot be written
        
        Examples:
            >>> exporter = ReportExporter()
            >>> html = exporter.export_to_html(report)
            >>> 
            >>> # Write to file
            >>> exporter.export_to_html(report, output_path="report.html")
            >>> 
            >>> # Use custom template
            >>> html = exporter.export_to_html(
            ...     report,
            ...     template_name="custom_report.html"
            ... )
        """
        if report is None:
            raise ValueError("Report cannot be None")
        
        try:
            # Determine template name if not specified
            if template_name is None:
                if isinstance(report, Report):
                    template_name = "report.html"
                else:  # Dashboard
                    template_name = "dashboard.html"
            
            # Try to use Jinja2 template if available
            if self.jinja_env:
                try:
                    template = self.jinja_env.get_template(template_name)
                    html = template.render(  # nosemgrep: python.flask.security.xss.audit.direct-use-of-jinja2.direct-use-of-jinja2 -- False positive: rendered through the autoescaped Environment above (select_autoescape) using packaged templates; no untrusted template source.
                        report=report,
                        data=report.model_dump(mode="json")
                    )
                except Exception:
                    # Fall back to basic HTML if template not found
                    html = self._generate_basic_html(report)
            else:
                # Generate basic HTML if Jinja2 not available
                html = self._generate_basic_html(report)
            
            # Write to file if path specified
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(html, encoding="utf-8")
            
            return html
            
        except Exception as e:
            raise ValueError(f"Failed to export to HTML: {str(e)}") from e
    
    def _generate_basic_html(self, report: Union[Report, Dashboard]) -> str:
        """
        Generate basic HTML without templates.
        
        Args:
            report: Report or Dashboard to convert
        
        Returns:
            Basic HTML string with inline CSS
        """
        if isinstance(report, Report):
            return self._generate_report_html(report)
        else:
            return self._generate_dashboard_html(report)
    
    def _generate_report_html(self, report: Report) -> str:
        """Generate HTML for a Report."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .header .meta {{
            opacity: 0.9;
            font-size: 0.9em;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .section h2 {{
            color: #667eea;
            margin-top: 0;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
        }}
        .dashboard {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .chart {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            border: 1px solid #e0e0e0;
        }}
        .chart h3 {{
            margin-top: 0;
            color: #555;
        }}
        .footer {{
            text-align: center;
            color: #888;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report.title}</h1>
        <div class="meta">
            <strong>Type:</strong> {report.report_type.value} | 
            <strong>Created:</strong> {report.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p>{report.summary}</p>
    </div>
"""
        
        # Add sections
        for section in report.sections:
            html += f"""
    <div class="section">
        <h2>{section['title']}</h2>
        <p>{section['content']}</p>
    </div>
"""
        
        # Add dashboard charts
        html += f"""
    <div class="dashboard">
        <h2>Dashboard: {report.dashboard.title}</h2>
"""
        if report.dashboard.description:
            html += f"        <p>{report.dashboard.description}</p>\n"
        
        for chart in report.dashboard.charts:
            html += f"""
        <div class="chart">
            <h3>{chart.title}</h3>
"""
            if chart.description:
                html += f"            <p>{chart.description}</p>\n"
            html += f"            <p><em>Chart Type:</em> {chart.chart_type.value}</p>\n"
            html += "        </div>\n"
        
        html += """    </div>
    
    <div class="footer">
        <p>Generated by Universal Agent Evaluation Framework (UAEF)</p>
    </div>
</body>
</html>
"""
        return html
    
    def _generate_dashboard_html(self, dashboard: Dashboard) -> str:
        """Generate HTML for a Dashboard."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dashboard.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .chart {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        .chart h2 {{
            color: #667eea;
            margin-top: 0;
            font-size: 1.3em;
        }}
        .footer {{
            text-align: center;
            color: #888;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{dashboard.title}</h1>
"""
        if dashboard.description:
            html += f"        <p>{dashboard.description}</p>\n"
        html += f"""        <p><small>Created: {dashboard.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</small></p>
    </div>
    
    <div class="charts-grid">
"""
        
        for chart in dashboard.charts:
            html += f"""
        <div class="chart">
            <h2>{chart.title}</h2>
"""
            if chart.description:
                html += f"            <p>{chart.description}</p>\n"
            html += f"            <p><strong>Type:</strong> {chart.chart_type.value}</p>\n"
            html += "        </div>\n"
        
        html += """    </div>
    
    <div class="footer">
        <p>Generated by Universal Agent Evaluation Framework (UAEF)</p>
    </div>
</body>
</html>
"""
        return html
    
    def export_to_pdf(
        self,
        report: Union[Report, Dashboard],
        output_path: Optional[Union[str, Path]] = None,
        use_weasyprint: bool = True
    ) -> bytes:
        """
        Export report or dashboard to PDF format.
        
        Supports two PDF generation methods:
        1. weasyprint (default): HTML-to-PDF conversion with CSS support
        2. reportlab: Direct PDF generation (fallback)
        
        Args:
            report: Report or Dashboard to export
            output_path: Optional file path to write PDF to
            use_weasyprint: Use weasyprint if available (default: True)
        
        Returns:
            PDF content as bytes
        
        Raises:
            ValueError: If report is None or invalid
            ImportError: If neither weasyprint nor reportlab is available
            IOError: If output_path is specified but file cannot be written
        
        Examples:
            >>> exporter = ReportExporter()
            >>> pdf_bytes = exporter.export_to_pdf(report)
            >>> 
            >>> # Write to file
            >>> exporter.export_to_pdf(report, output_path="report.pdf")
            >>> 
            >>> # Force reportlab usage
            >>> pdf_bytes = exporter.export_to_pdf(report, use_weasyprint=False)
        """
        if report is None:
            raise ValueError("Report cannot be None")
        
        try:
            # Try weasyprint first (better HTML/CSS support)
            if use_weasyprint:
                try:
                    from weasyprint import HTML
                    
                    # Generate HTML
                    html = self.export_to_html(report)
                    
                    # Convert to PDF
                    pdf_bytes = HTML(string=html).write_pdf()
                    
                    # Write to file if path specified
                    if output_path:
                        output_path = Path(output_path)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_bytes(pdf_bytes)
                    
                    return pdf_bytes
                    
                except ImportError:
                    # Fall back to reportlab
                    pass
            
            # Use reportlab as fallback
            try:
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
                from reportlab.lib.units import inch
                from reportlab.platypus import (
                    PageBreak,
                    Paragraph,
                    SimpleDocTemplate,
                    Spacer,
                    Table,
                    TableStyle,
                )
                
                # Create PDF buffer
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                story = []
                styles = getSampleStyleSheet()
                
                # Custom styles
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor=colors.HexColor('#667eea'),
                    spaceAfter=30
                )
                heading_style = ParagraphStyle(
                    'CustomHeading',
                    parent=styles['Heading2'],
                    fontSize=16,
                    textColor=colors.HexColor('#667eea'),
                    spaceAfter=12
                )
                
                # Add content based on type
                if isinstance(report, Report):
                    # Title
                    story.append(Paragraph(report.title, title_style))
                    story.append(Spacer(1, 0.2 * inch))
                    
                    # Metadata
                    meta_text = f"<b>Type:</b> {report.report_type.value} | <b>Created:</b> {report.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                    story.append(Paragraph(meta_text, styles['Normal']))
                    story.append(Spacer(1, 0.3 * inch))
                    
                    # Summary
                    story.append(Paragraph("Executive Summary", heading_style))
                    story.append(Paragraph(report.summary, styles['Normal']))
                    story.append(Spacer(1, 0.3 * inch))
                    
                    # Sections
                    for section in report.sections:
                        story.append(Paragraph(section['title'], heading_style))
                        story.append(Paragraph(section['content'], styles['Normal']))
                        story.append(Spacer(1, 0.2 * inch))
                    
                    # Dashboard info
                    story.append(PageBreak())
                    story.append(Paragraph(f"Dashboard: {report.dashboard.title}", heading_style))
                    if report.dashboard.description:
                        story.append(Paragraph(report.dashboard.description, styles['Normal']))
                    story.append(Spacer(1, 0.2 * inch))
                    
                    # Charts
                    for chart in report.dashboard.charts:
                        chart_text = f"<b>{chart.title}</b> ({chart.chart_type.value})"
                        story.append(Paragraph(chart_text, styles['Normal']))
                        if chart.description:
                            story.append(Paragraph(chart.description, styles['Normal']))
                        story.append(Spacer(1, 0.1 * inch))
                
                else:  # Dashboard
                    # Title
                    story.append(Paragraph(report.title, title_style))
                    story.append(Spacer(1, 0.2 * inch))
                    
                    if report.description:
                        story.append(Paragraph(report.description, styles['Normal']))
                        story.append(Spacer(1, 0.3 * inch))
                    
                    # Charts
                    for chart in report.charts:
                        story.append(Paragraph(chart.title, heading_style))
                        chart_info = f"<b>Type:</b> {chart.chart_type.value}"
                        story.append(Paragraph(chart_info, styles['Normal']))
                        if chart.description:
                            story.append(Paragraph(chart.description, styles['Normal']))
                        story.append(Spacer(1, 0.2 * inch))
                
                # Build PDF
                doc.build(story)
                pdf_bytes = buffer.getvalue()
                buffer.close()
                
                # Write to file if path specified
                if output_path:
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(pdf_bytes)
                
                return pdf_bytes
                
            except ImportError as e:
                raise ImportError(
                    "PDF export requires either 'weasyprint' or 'reportlab'. "
                    "Install with: pip install weasyprint  OR  pip install reportlab"
                ) from e
                
        except Exception as e:
            raise ValueError(f"Failed to export to PDF: {str(e)}") from e
    
    def export_to_csv(
        self,
        data: Union[Report, Dashboard],
        output_path: Optional[Union[str, Path]] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Export tabular data to CSV format.
        
        Extracts tabular data from charts and report sections. For reports,
        exports dimension scores, metric scores, and chart data. For dashboards,
        exports all chart data in tabular format.
        
        Args:
            data: Report or Dashboard to export
            output_path: Optional file path to write CSV to
            include_metadata: Include metadata rows at top (default: True)
        
        Returns:
            CSV string
        
        Raises:
            ValueError: If data is None or contains no tabular data
            IOError: If output_path is specified but file cannot be written
        
        Examples:
            >>> exporter = ReportExporter()
            >>> csv_str = exporter.export_to_csv(dashboard)
            >>> print(csv_str)
            Chart,Metric,Value
            Dimension Scores,Tool Calling,0.85
            Dimension Scores,Response Quality,0.92
            ...
            
            >>> # Write to file
            >>> exporter.export_to_csv(report, output_path="report.csv")
        """
        if data is None:
            raise ValueError("Data cannot be None")
        
        try:
            output = StringIO()
            writer = csv.writer(output)
            
            # Add metadata header if requested
            if include_metadata:
                if isinstance(data, Report):
                    writer.writerow(["Report Title", data.title])
                    writer.writerow(["Report Type", data.report_type.value])
                    writer.writerow(["Created", data.created_at.isoformat()])
                    writer.writerow([])  # Blank line
                else:  # Dashboard
                    writer.writerow(["Dashboard Title", data.title])
                    writer.writerow(["Created", data.created_at.isoformat()])
                    writer.writerow([])  # Blank line
            
            # Extract tabular data from charts
            dashboard = data.dashboard if isinstance(data, Report) else data
            
            for chart in dashboard.charts:
                # Write chart header
                writer.writerow([f"Chart: {chart.title}"])
                writer.writerow([f"Type: {chart.chart_type.value}"])
                
                # Extract data based on chart type
                chart_data = chart.data
                
                # Handle common data structures
                if "dimensions" in chart_data and "scores" in chart_data:
                    # Dimension scores
                    writer.writerow(["Dimension", "Score"])
                    for dim, score in zip(chart_data["dimensions"], chart_data["scores"]):
                        writer.writerow([dim, score])
                
                elif "metrics" in chart_data and "values" in chart_data:
                    # Metric values
                    writer.writerow(["Metric", "Value"])
                    for metric, value in zip(chart_data["metrics"], chart_data["values"]):
                        writer.writerow([metric, value])
                
                elif "timestamps" in chart_data and "values" in chart_data:
                    # Time series data
                    writer.writerow(["Timestamp", "Value"])
                    for ts, val in zip(chart_data["timestamps"], chart_data["values"]):
                        writer.writerow([ts, val])
                
                elif "matrix" in chart_data:
                    # Matrix data (heatmap)
                    matrix = chart_data["matrix"]
                    x_labels = chart_data.get("x_labels", [])
                    y_labels = chart_data.get("y_labels", [])
                    
                    # Write header row
                    header = [""] + x_labels if x_labels else [""]
                    writer.writerow(header)
                    
                    # Write data rows
                    for i, row in enumerate(matrix):
                        row_label = y_labels[i] if i < len(y_labels) else f"Row {i}"
                        writer.writerow([row_label] + row)
                
                elif "labels" in chart_data and "values" in chart_data:
                    # Generic labeled data
                    writer.writerow(["Label", "Value"])
                    for label, value in zip(chart_data["labels"], chart_data["values"]):
                        writer.writerow([label, value])
                
                else:
                    # Generic key-value pairs
                    writer.writerow(["Key", "Value"])
                    for key, value in chart_data.items():
                        if isinstance(value, (list, dict)):
                            value = json.dumps(value)
                        writer.writerow([key, value])
                
                writer.writerow([])  # Blank line between charts
            
            csv_str = output.getvalue()
            output.close()
            
            # Write to file if path specified
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(csv_str, encoding="utf-8")
            
            return csv_str
            
        except Exception as e:
            raise ValueError(f"Failed to export to CSV: {str(e)}") from e

    def export_to_markdown(
        self,
        data: Union[Report, Dashboard],
        output_path: Optional[Union[str, Path]] = None,
        include_metadata: bool = True,
    ) -> str:
        """
        Export a Report or Dashboard to Markdown format.

        Renders the summary, sections, and chart data as readable Markdown
        tables and bullet lists suitable for CI logs, PR comments, or docs.

        Args:
            data: Report or Dashboard to export
            output_path: Optional file path to write the Markdown to
            include_metadata: Include metadata header (default: True)

        Returns:
            Markdown string

        Raises:
            ValueError: If data is None

        Examples:
            >>> exporter = ReportExporter()
            >>> md = exporter.export_to_markdown(report)
            >>> print(md)

            >>> exporter.export_to_markdown(dashboard, output_path="report.md")
        """
        if data is None:
            raise ValueError("Data cannot be None")

        lines: list = []

        is_report = isinstance(data, Report)
        title = data.title
        dashboard = data.dashboard if is_report else data

        # Title
        lines.append(f"# {title}")
        lines.append("")

        # Metadata
        if include_metadata:
            created = data.created_at.strftime("%Y-%m-%d %H:%M UTC")
            lines.append(f"*Generated: {created}*")
            if is_report:
                lines.append(f"*Report type: {data.report_type.value}*")
            lines.append("")

        # Summary (reports only)
        if is_report and data.summary:
            lines.append("## Summary")
            lines.append("")
            lines.append(data.summary)
            lines.append("")

        # Sections (reports only)
        if is_report and data.sections:
            for section in data.sections:
                lines.append(f"## {section['title']}")
                lines.append("")
                lines.append(section["content"])
                lines.append("")

        # Charts as tables
        if dashboard.charts:
            lines.append("## Metrics")
            lines.append("")
            for chart in dashboard.charts:
                lines.append(f"### {chart.title}")
                if chart.description:
                    lines.append(f"_{chart.description}_")
                lines.append("")
                lines.extend(self._chart_to_markdown_table(chart))
                lines.append("")

        md_str = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(md_str, encoding="utf-8")

        return md_str

    @staticmethod
    def _chart_to_markdown_table(chart) -> list:
        """Convert a Chart's data dict into Markdown table lines."""
        lines: list = []
        chart_data = chart.data

        if "dimensions" in chart_data and "scores" in chart_data:
            lines.append("| Dimension | Score |")
            lines.append("|-----------|-------|")
            for dim, score in zip(chart_data["dimensions"], chart_data["scores"]):
                score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
                lines.append(f"| {dim} | {score_str} |")

        elif "metrics" in chart_data and ("values" in chart_data or "scores" in chart_data):
            values = chart_data.get("values", chart_data.get("scores", []))
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for metric, value in zip(chart_data["metrics"], values):
                val_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                lines.append(f"| {metric} | {val_str} |")

        elif "labels" in chart_data and "values" in chart_data:
            lines.append("| Label | Value |")
            lines.append("|-------|-------|")
            for label, value in zip(chart_data["labels"], chart_data["values"]):
                val_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                lines.append(f"| {label} | {val_str} |")

        elif "matrix" in chart_data:
            x_labels = chart_data.get("x_labels", [])
            y_labels = chart_data.get("y_labels", [])
            matrix = chart_data["matrix"]
            header = "| |" + "|".join(f" {x} " for x in x_labels) + "|"
            sep = "|---|" + "|".join("---" for _ in x_labels) + "|"
            lines.append(header)
            lines.append(sep)
            for i, row in enumerate(matrix):
                row_label = y_labels[i] if i < len(y_labels) else f"Row {i}"
                cells = "|".join(f" {v:.2f} " if isinstance(v, float) else f" {v} " for v in row)
                lines.append(f"| {row_label} |{cells}|")

        else:
            # Fallback: key-value list
            for key, value in chart_data.items():
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, default=str)
                lines.append(f"- **{key}**: {value}")

        return lines
