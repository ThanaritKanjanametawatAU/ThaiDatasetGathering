"""
Quality Report Generator for Audio Enhancement
Generates comprehensive reports in multiple formats with insights and visualizations
"""

import json
import os
import logging
import time
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from jinja2 import Template, Environment, FileSystemLoader
from .pdf_utils import (
    colors, letter, A4, SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer, Image, PageBreak, getSampleStyleSheet,
    ParagraphStyle, inch, TA_CENTER, TA_RIGHT, Drawing,
    LinePlot, VerticalBarChart, REPORTLAB_AVAILABLE
)
import io
import base64

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """Represents a section in the report"""
    title: str
    content: Any
    charts: List[str] = field(default_factory=list)
    tables: List[Dict] = field(default_factory=list)
    insights: List[Dict] = field(default_factory=list)


@dataclass
class ReportTemplate:
    """Defines report template structure"""
    name: str
    sections: List[str]
    style: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisualizationEngine:
    """Handles all visualization generation for reports"""
    
    def __init__(self, style_config: Optional[Dict] = None):
        # Merge provided style with defaults
        default_style = self._get_default_style()
        if style_config:
            default_style.update(style_config)
        self.style_config = default_style
        self._setup_plotting_style()
    
    def _get_default_style(self) -> Dict:
        """Get default visualization style"""
        return {
            'color_palette': ['#0066CC', '#66B2FF', '#FFB366', '#FF6666', '#66FF99'],
            'figure_size': (10, 6),
            'dpi': 100,
            'font_size': 12,
            'grid': True
        }
    
    def _setup_plotting_style(self):
        """Setup matplotlib and seaborn styles"""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            # Fallback to default style if seaborn style not available
            plt.style.use('default')
        
        if 'color_palette' in self.style_config:
            sns.set_palette(self.style_config['color_palette'])
        
        plt.rcParams['figure.figsize'] = self.style_config.get('figure_size', (10, 6))
        plt.rcParams['figure.dpi'] = self.style_config.get('dpi', 100)
        plt.rcParams['font.size'] = self.style_config.get('font_size', 12)
    
    def create_distribution_plot(self, data: np.ndarray, title: str, 
                               metric_name: str) -> Tuple[plt.Figure, str]:
        """Create distribution plot for metrics"""
        fig, ax = plt.subplots(figsize=self.style_config['figure_size'])
        
        # Create histogram with KDE
        sns.histplot(data, kde=True, ax=ax, color=self.style_config['color_palette'][0])
        
        # Add statistics
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Add text box with statistics
        textstr = f'μ = {mean_val:.2f}\nσ = {std_val:.2f}\nn = {len(data)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        # Convert to base64 for embedding
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.style_config['dpi'])
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return fig, image_base64
    
    def create_trend_plot(self, dates: pd.DatetimeIndex, values: np.ndarray, 
                         title: str) -> Tuple[plt.Figure, str]:
        """Create trend plot over time"""
        fig, ax = plt.subplots(figsize=self.style_config['figure_size'])
        
        # Plot trend line
        ax.plot(dates, values, linewidth=2, color=self.style_config['color_palette'][0], 
                label='Actual')
        
        # Add moving average
        window = min(7, len(values) // 3)
        if window > 1:
            ma = pd.Series(values).rolling(window=window, center=True).mean()
            ax.plot(dates, ma, linewidth=2, color=self.style_config['color_palette'][1], 
                    label=f'{window}-day MA', linestyle='--')
        
        # Add trend line
        z = np.polyfit(range(len(dates)), values, 1)
        p = np.poly1d(z)
        ax.plot(dates, p(range(len(dates))), linewidth=2, 
                color=self.style_config['color_palette'][2], 
                label='Trend', linestyle=':')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.style_config['dpi'])
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return fig, image_base64
    
    def create_comparison_plot(self, methods: List[str], 
                             metrics: Dict[str, List[float]]) -> Tuple[plt.Figure, str]:
        """Create comparison bar plot"""
        fig, ax = plt.subplots(figsize=self.style_config['figure_size'])
        
        x = np.arange(len(methods))
        width = 0.8 / len(metrics)
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            offset = (i - len(metrics)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, 
                          label=metric_name, 
                          color=self.style_config['color_palette'][i % len(self.style_config['color_palette'])])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Methods')
        ax.set_ylabel('Score')
        ax.set_title('Method Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.style_config['dpi'])
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return fig, image_base64
    
    def create_heatmap(self, data: np.ndarray, labels: List[str], 
                      title: str) -> Tuple[plt.Figure, str]:
        """Create correlation heatmap"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm', 
                   xticklabels=labels, yticklabels=labels, 
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.style_config['dpi'])
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return fig, image_base64
    
    def create_interactive_dashboard(self, metrics_data: Dict) -> str:
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PESQ Distribution', 'Quality Trend', 
                          'Method Comparison', 'Processing Volume'),
            specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Add distribution plot
        if 'distributions' in metrics_data and 'pesq' in metrics_data['distributions']:
            fig.add_trace(
                go.Histogram(x=metrics_data['distributions']['pesq'], 
                           name='PESQ Distribution'),
                row=1, col=1
            )
        
        # Add trend plot
        if 'trends' in metrics_data:
            fig.add_trace(
                go.Scatter(x=metrics_data['trends'].get('dates', []), 
                         y=metrics_data['trends'].get('pesq_trend', []),
                         mode='lines+markers', name='PESQ Trend'),
                row=1, col=2
            )
        
        # Add comparison plot
        if 'comparisons' in metrics_data:
            fig.add_trace(
                go.Bar(x=metrics_data['comparisons'].get('methods', []),
                      y=metrics_data['comparisons'].get('pesq_scores', []),
                      name='PESQ Scores'),
                row=2, col=1
            )
        
        # Add volume plot
        if 'trends' in metrics_data and 'volume' in metrics_data['trends']:
            fig.add_trace(
                go.Scatter(x=metrics_data['trends'].get('dates', []),
                         y=metrics_data['trends'].get('volume', []),
                         mode='lines', fill='tozeroy', name='Processing Volume'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Audio Quality Dashboard")
        
        # Convert to HTML
        return fig.to_html(include_plotlyjs='cdn')


class InsightGenerator:
    """Generates automated insights from metrics data"""
    
    def __init__(self):
        self.insight_rules = self._load_insight_rules()
    
    def _load_insight_rules(self) -> Dict:
        """Load rules for insight generation"""
        return {
            'quality_thresholds': {
                'pesq': {'poor': 2.0, 'fair': 2.5, 'good': 3.0, 'excellent': 3.5},
                'stoi': {'poor': 0.6, 'fair': 0.7, 'good': 0.8, 'excellent': 0.9},
                'si_sdr': {'poor': 5.0, 'fair': 10.0, 'good': 15.0, 'excellent': 20.0}
            },
            'trend_thresholds': {
                'improvement': 0.05,  # 5% improvement
                'degradation': -0.05  # 5% degradation
            },
            'issue_thresholds': {
                'high_failure_rate': 0.1,  # 10% failures
                'low_quality_rate': 0.2   # 20% low quality
            }
        }
    
    def generate_insights(self, metrics_data: Dict) -> List[Dict]:
        """Generate insights from metrics data"""
        insights = []
        
        # Analyze summary metrics
        if 'summary' in metrics_data:
            insights.extend(self._analyze_summary(metrics_data['summary']))
        
        # Analyze trends
        if 'trends' in metrics_data:
            insights.extend(self._analyze_trends(metrics_data['trends']))
        
        # Analyze issues
        if 'issues' in metrics_data:
            insights.extend(self._analyze_issues(metrics_data['issues']))
        
        # Analyze distributions
        if 'distributions' in metrics_data:
            insights.extend(self._analyze_distributions(metrics_data['distributions']))
        
        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}
        insights.sort(key=lambda x: severity_order.get(x['severity'], 5))
        
        return insights
    
    def _analyze_summary(self, summary: Dict) -> List[Dict]:
        """Analyze summary metrics"""
        insights = []
        
        # Check PESQ quality
        if 'average_pesq' in summary:
            pesq = summary['average_pesq']
            thresholds = self.insight_rules['quality_thresholds']['pesq']
            
            if pesq < thresholds['poor']:
                insights.append({
                    'type': 'quality_issue',
                    'severity': 'critical',
                    'message': f'Average PESQ score ({pesq:.2f}) is below poor quality threshold',
                    'recommendation': 'Review enhancement parameters and consider more aggressive noise reduction'
                })
            elif pesq < thresholds['fair']:
                insights.append({
                    'type': 'quality_issue',
                    'severity': 'high',
                    'message': f'Average PESQ score ({pesq:.2f}) indicates poor quality',
                    'recommendation': 'Analyze failing samples and adjust enhancement strategies'
                })
            elif pesq > thresholds['excellent']:
                insights.append({
                    'type': 'quality_achievement',
                    'severity': 'info',
                    'message': f'Excellent average PESQ score ({pesq:.2f})',
                    'recommendation': 'Current enhancement settings are performing well'
                })
        
        # Check success rate
        if 'success_rate' in summary:
            success_rate = summary['success_rate']
            if success_rate < 0.9:
                insights.append({
                    'type': 'processing_issue',
                    'severity': 'high',
                    'message': f'Low success rate ({success_rate:.1%})',
                    'recommendation': 'Investigate processing failures and improve error handling'
                })
        
        return insights
    
    def _analyze_trends(self, trends: Dict) -> List[Dict]:
        """Analyze quality trends"""
        insights = []
        
        # Check PESQ trend
        if 'pesq_trend' in trends:
            values = trends['pesq_trend']
            if len(values) > 7:
                # Calculate week-over-week change
                recent = np.mean(values[-7:])
                previous = np.mean(values[-14:-7])
                change = (recent - previous) / previous if previous > 0 else 0
                
                if change > self.insight_rules['trend_thresholds']['improvement']:
                    insights.append({
                        'type': 'quality_trend',
                        'severity': 'info',
                        'message': f'Quality improving: {change:.1%} increase in last week',
                        'recommendation': 'Continue current approach and monitor for consistency'
                    })
                elif change < self.insight_rules['trend_thresholds']['degradation']:
                    insights.append({
                        'type': 'quality_trend',
                        'severity': 'medium',
                        'message': f'Quality degrading: {change:.1%} decrease in last week',
                        'recommendation': 'Review recent changes and identify degradation causes'
                    })
        
        return insights
    
    def _analyze_issues(self, issues: Dict) -> List[Dict]:
        """Analyze processing issues"""
        insights = []
        
        total_issues = sum(issues.values())
        
        if total_issues > 0:
            # Check low quality rate
            if 'low_quality' in issues:
                low_quality_rate = issues['low_quality'] / total_issues
                if low_quality_rate > self.insight_rules['issue_thresholds']['low_quality_rate']:
                    insights.append({
                        'type': 'quality_issue',
                        'severity': 'high',
                        'message': f'High rate of low quality outputs ({low_quality_rate:.1%})',
                        'recommendation': 'Review quality thresholds and enhancement parameters'
                    })
            
            # Check failure rate
            if 'processing_failures' in issues:
                failure_rate = issues['processing_failures'] / total_issues
                if failure_rate > self.insight_rules['issue_thresholds']['high_failure_rate']:
                    insights.append({
                        'type': 'processing_issue',
                        'severity': 'critical',
                        'message': f'High processing failure rate ({failure_rate:.1%})',
                        'recommendation': 'Debug processing pipeline and improve error handling'
                    })
        
        return insights
    
    def _analyze_distributions(self, distributions: Dict) -> List[Dict]:
        """Analyze metric distributions"""
        insights = []
        
        # Check for bimodal distributions
        for metric, values in distributions.items():
            if isinstance(values, np.ndarray) and len(values) > 100:
                # Simple bimodality check using histogram
                hist, bins = np.histogram(values, bins=20)
                peaks = self._find_peaks(hist)
                
                if len(peaks) > 1:
                    insights.append({
                        'type': 'distribution_issue',
                        'severity': 'medium',
                        'message': f'{metric} shows bimodal distribution, indicating inconsistent quality',
                        'recommendation': 'Investigate different quality clusters and standardize processing'
                    })
        
        return insights
    
    def _find_peaks(self, hist: np.ndarray) -> List[int]:
        """Simple peak detection in histogram"""
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                if hist[i] > np.max(hist) * 0.3:  # Significant peaks only
                    peaks.append(i)
        return peaks


class QualityReportGenerator:
    """Main class for generating quality reports"""
    
    # Predefined templates
    TEMPLATES = {
        'standard': ReportTemplate(
            name='standard',
            sections=['summary', 'distributions', 'trends', 'comparisons', 'insights', 'recommendations'],
            style={'font_family': 'Arial', 'primary_color': '#0066CC', 'page_size': 'letter'}
        ),
        'executive_summary': ReportTemplate(
            name='executive_summary',
            sections=['summary', 'key_metrics', 'insights', 'recommendations'],
            style={'font_family': 'Helvetica', 'primary_color': '#003366', 'page_size': 'letter'}
        ),
        'technical_detailed': ReportTemplate(
            name='technical_detailed',
            sections=['summary', 'distributions', 'correlations', 'trends', 'comparisons', 
                     'detailed_analysis', 'insights', 'recommendations', 'appendix'],
            style={'font_family': 'Times', 'primary_color': '#000000', 'page_size': 'A4'}
        )
    }
    
    def __init__(self, template: Union[str, ReportTemplate] = 'standard'):
        """Initialize report generator"""
        if isinstance(template, str):
            self.template = self.TEMPLATES.get(template, self.TEMPLATES['standard'])
            self.template_name = template
        else:
            self.template = template
            self.template_name = template.name
        
        self.visualizer = VisualizationEngine(self.template.style)
        self.insight_generator = InsightGenerator()
        self.sections = {}
        self.metadata = {
            'generated_at': None,
            'format': None,
            'sections_generated': 0,
            'warnings': []
        }
        self.language = 'en'
        self.accessibility_enabled = False
        self._scheduler = None
    
    def generate_report(self, metrics_data: Dict, output_path: str, 
                       format: str = 'pdf') -> str:
        """Generate quality report in specified format"""
        start_time = time.time()
        
        # Validate format
        if format not in ['pdf', 'html', 'json']:
            raise ValueError(f"Unsupported format: {format}")
        
        # Update metadata
        self.metadata['generated_at'] = datetime.now().isoformat()
        self.metadata['format'] = format
        
        # Validate and prepare data
        metrics_data = self._validate_metrics_data(metrics_data)
        
        # Generate insights
        if 'insights' not in metrics_data or not metrics_data['insights']:
            metrics_data['insights'] = self.insight_generator.generate_insights(metrics_data)
        
        # Generate report based on format
        if format == 'pdf':
            output_path = self._generate_pdf_report(metrics_data, output_path)
        elif format == 'html':
            output_path = self._generate_html_report(metrics_data, output_path)
        elif format == 'json':
            output_path = self._generate_json_report(metrics_data, output_path)
        
        # Update metadata
        self.metadata['generation_time'] = time.time() - start_time
        if self.metadata['sections_generated'] == 0:
            self.metadata['sections_generated'] = len(self.sections) if self.sections else len(self.template.sections)
        self.metadata['memory_usage_mb'] = psutil.Process().memory_info().rss / 1024 / 1024
        
        logger.info(f"Report generated successfully: {output_path}")
        return output_path
    
    def _validate_metrics_data(self, metrics_data: Dict) -> Dict:
        """Validate and prepare metrics data"""
        # Ensure required sections exist
        if 'summary' not in metrics_data:
            self.metadata['warnings'].append('Missing summary section')
            metrics_data['summary'] = {}
        
        # Check for empty summary
        if not metrics_data.get('summary'):
            if 'Missing summary section' not in self.metadata['warnings']:
                self.metadata['warnings'].append('Summary section is empty')
        
        # Convert numpy arrays to lists for JSON serialization
        if 'distributions' in metrics_data:
            for key, value in metrics_data['distributions'].items():
                if isinstance(value, np.ndarray):
                    metrics_data['distributions'][key] = value.tolist()
        
        return metrics_data
    
    def _generate_pdf_report(self, metrics_data: Dict, output_path: str) -> str:
        """Generate PDF report"""
        # Ensure .pdf extension
        if not output_path.endswith('.pdf'):
            output_path += '.pdf'
        
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter if self.template.style.get('page_size') == 'letter' else A4,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=18
        )
        
        # Build story
        story = []
        styles = getSampleStyleSheet()
        
        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor(self.template.style.get('primary_color', '#000000')),
            alignment=TA_CENTER
        )
        story.append(Paragraph("Audio Enhancement Quality Report", title_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Add generation date
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                             styles['Normal']))
        story.append(Spacer(1, 0.25*inch))
        
        # Add sections based on template
        sections_added = 0
        for section_name in self.template.sections:
            if section_name == 'summary':
                story.extend(self._create_pdf_summary_section(metrics_data, styles))
                sections_added += 1
            elif section_name == 'distributions':
                story.extend(self._create_pdf_distributions_section(metrics_data, styles))
                sections_added += 1
            elif section_name == 'trends':
                story.extend(self._create_pdf_trends_section(metrics_data, styles))
                sections_added += 1
            elif section_name == 'insights':
                story.extend(self._create_pdf_insights_section(metrics_data, styles))
                sections_added += 1
            
            story.append(PageBreak())
        
        self.metadata['sections_generated'] = sections_added
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def _create_pdf_summary_section(self, metrics_data: Dict, styles) -> List:
        """Create summary section for PDF"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", styles['Heading1']))
        elements.append(Spacer(1, 0.25*inch))
        
        if 'summary' in metrics_data:
            summary = metrics_data['summary']
            
            # Create summary table
            data = [['Metric', 'Value']]
            if 'total_samples' in summary:
                data.append(['Total Samples', f"{summary['total_samples']:,}"])
            if 'average_pesq' in summary:
                data.append(['Average PESQ', f"{summary['average_pesq']:.2f}"])
            if 'average_stoi' in summary:
                data.append(['Average STOI', f"{summary['average_stoi']:.3f}"])
            if 'average_si_sdr' in summary:
                data.append(['Average SI-SDR', f"{summary['average_si_sdr']:.1f} dB"])
            if 'success_rate' in summary:
                data.append(['Success Rate', f"{summary['success_rate']:.1%}"])
            
            table = Table(data, colWidths=[3*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.template.style.get('primary_color', '#0066CC'))),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(table)
        
        return elements
    
    def _create_pdf_distributions_section(self, metrics_data: Dict, styles) -> List:
        """Create distributions section for PDF"""
        elements = []
        
        elements.append(Paragraph("Metric Distributions", styles['Heading1']))
        elements.append(Spacer(1, 0.25*inch))
        
        if 'distributions' in metrics_data:
            for metric, values in metrics_data['distributions'].items():
                if isinstance(values, (list, np.ndarray)):
                    # Create distribution plot
                    fig, image_base64 = self.visualizer.create_distribution_plot(
                        np.array(values), f"{metric.upper()} Distribution", metric.upper()
                    )
                    
                    # Add image to PDF
                    img_buffer = io.BytesIO(base64.b64decode(image_base64))
                    img = Image(img_buffer, width=5*inch, height=3*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.25*inch))
        
        return elements
    
    def _create_pdf_trends_section(self, metrics_data: Dict, styles) -> List:
        """Create trends section for PDF"""
        elements = []
        
        elements.append(Paragraph("Quality Trends", styles['Heading1']))
        elements.append(Spacer(1, 0.25*inch))
        
        if 'trends' in metrics_data:
            if 'dates' in metrics_data['trends'] and 'pesq_trend' in metrics_data['trends']:
                # Create trend plot
                fig, image_base64 = self.visualizer.create_trend_plot(
                    metrics_data['trends']['dates'],
                    metrics_data['trends']['pesq_trend'],
                    "PESQ Trend Over Time"
                )
                
                # Add image to PDF
                img_buffer = io.BytesIO(base64.b64decode(image_base64))
                img = Image(img_buffer, width=5*inch, height=3*inch)
                elements.append(img)
        
        return elements
    
    def _create_pdf_insights_section(self, metrics_data: Dict, styles) -> List:
        """Create insights section for PDF"""
        elements = []
        
        elements.append(Paragraph("Key Insights", styles['Heading1']))
        elements.append(Spacer(1, 0.25*inch))
        
        if 'insights' in metrics_data:
            for insight in metrics_data['insights']:
                # Add severity icon
                severity_colors = {
                    'critical': colors.red,
                    'high': colors.orange,
                    'medium': colors.yellow,
                    'low': colors.green,
                    'info': colors.blue
                }
                
                # Create insight paragraph
                severity_style = ParagraphStyle(
                    'InsightStyle',
                    parent=styles['Normal'],
                    textColor=severity_colors.get(insight['severity'], colors.black),
                    fontName='Helvetica-Bold'
                )
                
                elements.append(Paragraph(f"[{insight['severity'].upper()}] {insight['message']}", 
                                        severity_style))
                elements.append(Paragraph(f"Recommendation: {insight['recommendation']}", 
                                        styles['Italic']))
                elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def _generate_html_report(self, metrics_data: Dict, output_path: str) -> str:
        """Generate HTML report"""
        # Ensure .html extension
        if not output_path.endswith('.html'):
            output_path += '.html'
        
        # Create HTML template
        html_template = """
        <!DOCTYPE html>
        <html lang="{{ language }}">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Audio Enhancement Quality Report</title>
            <style>
                body {
                    font-family: {{ font_family }};
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f4f4f4;
                }
                h1, h2, h3 {
                    color: {{ primary_color }};
                }
                .summary-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .summary-table th, .summary-table td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                .summary-table th {
                    background-color: {{ primary_color }};
                    color: white;
                }
                .chart-container {
                    margin: 20px 0;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .insight {
                    padding: 10px;
                    margin: 10px 0;
                    border-left: 4px solid;
                    background-color: white;
                }
                .insight.critical { border-color: #dc3545; }
                .insight.high { border-color: #fd7e14; }
                .insight.medium { border-color: #ffc107; }
                .insight.low { border-color: #28a745; }
                .insight.info { border-color: #17a2b8; }
                {%- if accessibility_enabled %}
                /* Accessibility enhancements */
                a:focus, button:focus {
                    outline: 3px solid #0066CC;
                    outline-offset: 2px;
                }
                img {
                    max-width: 100%;
                    height: auto;
                }
                {%- endif %}
            </style>
        </head>
        <body>
            <h1>Audio Enhancement Quality Report</h1>
            <p>Generated: {{ generation_date }}</p>
            
            <section id="summary" role="region" aria-label="Summary"{% if accessibility_enabled %} tabindex="0"{% endif %}>
                <h2>Executive Summary</h2>
                <table class="summary-table" role="table"{% if accessibility_enabled %} tabindex="0"{% endif %}>
                    <thead>
                        <tr>
                            <th scope="col">Metric</th>
                            <th scope="col">Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for key, value in summary.items() %}
                        <tr>
                            <td>{{ key | title | replace('_', ' ') }}</td>
                            <td>{{ value }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </section>
            
            {% if charts %}
            <section id="visualizations" role="region" aria-label="Visualizations">
                <h2>Visualizations</h2>
                {% for chart in charts %}
                <div class="chart-container">
                    <h3>{{ chart.title }}</h3>
                    <img src="data:image/png;base64,{{ chart.image }}" 
                         alt="{{ chart.alt_text }}"
                         {%- if accessibility_enabled %} role="img"{% endif %}>
                </div>
                {% endfor %}
            </section>
            {% endif %}
            
            {% if insights %}
            <section id="insights" role="region" aria-label="Insights">
                <h2>Key Insights</h2>
                {% for insight in insights %}
                <div class="insight {{ insight.severity }}" role="alert">
                    <strong>[{{ insight.severity | upper }}]</strong> {{ insight.message }}
                    <br>
                    <em>Recommendation:</em> {{ insight.recommendation }}
                </div>
                {% endfor %}
            </section>
            {% endif %}
            
            {% if interactive_dashboard %}
            <section id="dashboard" role="region" aria-label="Interactive Dashboard">
                <h2>Interactive Dashboard</h2>
                {{ interactive_dashboard | safe }}
            </section>
            {% endif %}
        </body>
        </html>
        """
        
        # Prepare template data
        template_data = {
            'language': self.language,
            'font_family': self.template.style.get('font_family', 'Arial'),
            'primary_color': self.template.style.get('primary_color', '#0066CC'),
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': metrics_data.get('summary', {}),
            'charts': [],
            'insights': metrics_data.get('insights', []),
            'accessibility_enabled': self.accessibility_enabled
        }
        
        # Generate charts
        if 'distributions' in metrics_data:
            for metric, values in metrics_data['distributions'].items():
                if isinstance(values, (list, np.ndarray)):
                    fig, image_base64 = self.visualizer.create_distribution_plot(
                        np.array(values), f"{metric.upper()} Distribution", metric.upper()
                    )
                    template_data['charts'].append({
                        'title': f"{metric.upper()} Distribution",
                        'image': image_base64,
                        'alt_text': f"Histogram showing distribution of {metric} values"
                    })
        
        # Add interactive dashboard
        if 'trends' in metrics_data or 'comparisons' in metrics_data:
            template_data['interactive_dashboard'] = self.visualizer.create_interactive_dashboard(metrics_data)
        
        # Render template
        template = Template(html_template)
        html_content = template.render(**template_data)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_json_report(self, metrics_data: Dict, output_path: str) -> str:
        """Generate JSON report"""
        # Ensure .json extension
        if not output_path.endswith('.json'):
            output_path += '.json'
        
        # Prepare JSON data
        json_data = {
            'metadata': {
                'generated_at': self.metadata['generated_at'],
                'template': self.template_name,
                'version': '1.0',
                'generator': 'QualityReportGenerator'
            },
            'summary': metrics_data.get('summary', {}),
            'distributions': {},
            'trends': {},
            'comparisons': metrics_data.get('comparisons', {}),
            'insights': metrics_data.get('insights', []),
            'issues': metrics_data.get('issues', {})
        }
        
        # Convert distributions
        if 'distributions' in metrics_data:
            for key, value in metrics_data['distributions'].items():
                if isinstance(value, np.ndarray):
                    json_data['distributions'][key] = {
                        'values': value.tolist(),
                        'statistics': {
                            'mean': float(np.mean(value)),
                            'std': float(np.std(value)),
                            'min': float(np.min(value)),
                            'max': float(np.max(value)),
                            'median': float(np.median(value))
                        }
                    }
                else:
                    json_data['distributions'][key] = value
        
        # Convert trends
        if 'trends' in metrics_data:
            json_data['trends'] = {}
            for key, value in metrics_data['trends'].items():
                if isinstance(value, pd.DatetimeIndex):
                    json_data['trends'][key] = [d.isoformat() for d in value]
                elif isinstance(value, np.ndarray):
                    json_data['trends'][key] = value.tolist()
                else:
                    json_data['trends'][key] = value
        
        # Write JSON file
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return output_path
    
    def generate_insights(self, metrics_data: Dict) -> List[Dict]:
        """Generate insights from metrics data"""
        return self.insight_generator.generate_insights(metrics_data)
    
    def add_section(self, section_name: str, section: ReportSection):
        """Add custom section to report"""
        self.sections[section_name] = section
        if section_name not in self.template.sections:
            self.template.sections.append(section_name)
    
    def get_sections(self) -> List[str]:
        """Get list of report sections"""
        return self.template.sections + list(self.sections.keys())
    
    def generate_batch_reports(self, batch_data: Dict[str, Dict], 
                             output_dir: str, format: str = 'html') -> Dict[str, str]:
        """Generate multiple reports in batch"""
        report_paths = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset_name, metrics_data in batch_data.items():
            output_path = os.path.join(output_dir, f"{dataset_name}_report.{format}")
            report_path = self.generate_report(metrics_data, output_path, format)
            report_paths[dataset_name] = report_path
        
        return report_paths
    
    def get_report_metadata(self) -> Dict:
        """Get report generation metadata"""
        return self.metadata.copy()
    
    def set_language(self, language: str):
        """Set report language"""
        self.language = language
    
    def enable_accessibility_features(self):
        """Enable WCAG accessibility features"""
        self.accessibility_enabled = True
    
    def schedule_reports(self, frequency: str, time: str, 
                        metrics_source: str, output_dir: str):
        """Schedule periodic report generation"""
        # This is a placeholder for scheduling functionality
        # In real implementation, would use APScheduler or similar
        logger.info(f"Scheduling {frequency} reports at {time}")
        pass
    
    def schedule_triggered_reports(self, trigger: str, condition: Dict, 
                                  output_format: str):
        """Schedule triggered report generation"""
        # This is a placeholder for triggered report functionality
        logger.info(f"Scheduling triggered reports on {trigger} with condition {condition}")
        pass
    
    def generate_comparison_report(self, baseline: Dict, current: Dict, 
                                  output_path: str) -> Dict:
        """Generate comparison report between two periods"""
        comparison = {
            'improvements': {},
            'degradations': {},
            'overall_trend': 'stable'
        }
        
        # Compare summary metrics
        if 'summary' in baseline and 'summary' in current:
            for metric in ['average_pesq', 'average_stoi', 'average_si_sdr']:
                if metric in baseline['summary'] and metric in current['summary']:
                    baseline_val = baseline['summary'][metric]
                    current_val = current['summary'][metric]
                    improvement = current_val - baseline_val
                    
                    if improvement > 0:
                        comparison['improvements'][metric.replace('average_', '')] = improvement
                    else:
                        comparison['degradations'][metric.replace('average_', '')] = abs(improvement)
        
        # Determine overall trend
        if len(comparison['improvements']) > len(comparison['degradations']):
            comparison['overall_trend'] = 'improving'
        elif len(comparison['degradations']) > len(comparison['improvements']):
            comparison['overall_trend'] = 'degrading'
        
        # Generate comparison report
        comparison_data = {
            'summary': {
                'baseline_period': baseline.get('period', 'Baseline'),
                'current_period': current.get('period', 'Current'),
                'overall_trend': comparison['overall_trend']
            },
            'comparisons': comparison,
            'baseline_metrics': baseline,
            'current_metrics': current
        }
        
        self.generate_report(comparison_data, output_path, format='html')
        
        return comparison