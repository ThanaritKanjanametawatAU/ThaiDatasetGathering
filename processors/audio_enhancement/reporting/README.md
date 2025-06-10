# Audio Enhancement Quality Report Generator

## Overview

The Quality Report Generator is a comprehensive system for creating detailed, actionable reports on audio enhancement quality metrics, trends, and recommendations. It supports multiple output formats (PDF, HTML, JSON) and includes automated insight generation.

## Features

### Core Features
- **Multi-format report generation**: PDF, HTML, and JSON formats
- **Customizable report templates**: Standard, Executive Summary, Technical Detailed
- **Automated insights and recommendations**: AI-powered analysis of metrics
- **Batch report generation**: Process multiple datasets at once
- **Interactive visualizations**: Charts, graphs, and heatmaps
- **WCAG accessibility compliance**: For HTML reports
- **Multi-language support**: Internationalization ready

### Report Types
1. **Executive Summary**: High-level overview for stakeholders
2. **Standard Report**: Balanced detail for most use cases  
3. **Technical Detailed**: Comprehensive analysis for engineers

### Visualization Components
- Distribution plots with statistics
- Time series trend analysis
- Method comparison charts
- Correlation heatmaps
- Interactive dashboards (HTML)

## Installation

The module requires these additional dependencies:
```bash
pip install reportlab jinja2 plotly
```

Note: If reportlab is not available, PDF generation will use a mock implementation with limited functionality.

## Usage

### Basic Example

```python
from processors.audio_enhancement.reporting import QualityReportGenerator

# Initialize generator
generator = QualityReportGenerator(template='standard')

# Prepare metrics data
metrics_data = {
    'summary': {
        'total_samples': 1000,
        'average_pesq': 3.2,
        'average_stoi': 0.87,
        'average_si_sdr': 15.4,
        'success_rate': 0.95
    },
    'distributions': {
        'pesq': pesq_scores_array,
        'stoi': stoi_scores_array
    },
    'trends': {
        'dates': date_range,
        'pesq_trend': pesq_over_time,
        'volume': processing_volume
    }
}

# Generate report
report_path = generator.generate_report(
    metrics_data,
    output_path='quality_report.pdf',
    format='pdf'
)
```

### Custom Templates

```python
from processors.audio_enhancement.reporting import ReportTemplate

# Create custom template
custom_template = ReportTemplate(
    name='minimal',
    sections=['summary', 'key_metrics', 'recommendations'],
    style={
        'font_family': 'Helvetica',
        'primary_color': '#0066CC',
        'page_size': 'A4'
    }
)

# Use custom template
generator = QualityReportGenerator(template=custom_template)
```

### Batch Processing

```python
# Process multiple datasets
batch_data = {
    'dataset_1': metrics_data_1,
    'dataset_2': metrics_data_2,
    'dataset_3': metrics_data_3
}

report_paths = generator.generate_batch_reports(
    batch_data,
    output_dir='reports/',
    format='html'
)
```

### Accessibility Features

```python
# Enable WCAG compliance for HTML reports
generator.enable_accessibility_features()
report_path = generator.generate_report(
    metrics_data,
    output_path='accessible_report.html',
    format='html'
)
```

## Metrics Data Structure

The generator expects metrics data in this format:

```python
{
    'summary': {
        'total_samples': int,
        'average_pesq': float,
        'average_stoi': float,
        'average_si_sdr': float,
        'processing_time': float,
        'success_rate': float
    },
    'distributions': {
        'metric_name': np.ndarray or list
    },
    'trends': {
        'dates': pd.DatetimeIndex or list,
        'metric_trend': np.ndarray or list
    },
    'comparisons': {
        'methods': list[str],
        'metric_scores': list[float]
    },
    'issues': {
        'issue_type': count
    }
}
```

## Automated Insights

The insight generator analyzes metrics and provides:

- **Quality assessments**: Based on PESQ, STOI, SI-SDR thresholds
- **Trend analysis**: Week-over-week improvements or degradations
- **Issue detection**: High failure rates, quality problems
- **Distribution analysis**: Bimodal distributions, outliers
- **Recommendations**: Actionable steps to improve quality

Insights are categorized by severity:
- **Critical**: Immediate action required
- **High**: Significant issues to address
- **Medium**: Notable concerns
- **Low**: Minor issues
- **Info**: General observations

## Performance

- Single report generation: < 5 seconds
- Batch of 10 reports: < 30 seconds  
- Memory usage: < 500MB typical
- Chart rendering: < 500ms each

## Integration

The Quality Report Generator integrates with:
- Audio enhancement pipeline metrics
- Quality monitoring system
- Evaluation framework
- Dashboard systems

## Example Reports

See `examples/test_quality_report_generator.py` for a complete demonstration that generates:
- Standard quality reports in all formats
- Executive summary reports
- Batch reports for multiple datasets
- Accessible HTML reports

## Future Enhancements

- Real-time report generation API
- Report scheduling and automation
- Cloud storage integration
- Advanced ML-based insights
- Report comparison and diff views
- Email/webhook notifications