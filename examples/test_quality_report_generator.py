#!/usr/bin/env python3
"""
Example script demonstrating Quality Report Generator usage
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.reporting import QualityReportGenerator


def generate_sample_metrics():
    """Generate sample metrics data for demonstration"""
    # Generate realistic looking metrics
    np.random.seed(42)
    n_samples = 1000
    
    # Create distributions
    pesq_scores = np.clip(np.random.normal(3.2, 0.5, n_samples), 1.0, 4.5)
    stoi_scores = np.clip(np.random.normal(0.87, 0.1, n_samples), 0.0, 1.0)
    si_sdr_scores = np.random.normal(15.4, 3.0, n_samples)
    
    # Create trends over 30 days
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    pesq_trend = 2.8 + np.cumsum(np.random.normal(0.02, 0.05, 30))  # Gradual improvement
    stoi_trend = 0.80 + np.cumsum(np.random.normal(0.002, 0.01, 30))
    volume_trend = np.random.randint(80, 120, 30)
    
    return {
        'summary': {
            'total_samples': n_samples,
            'average_pesq': float(np.mean(pesq_scores)),
            'average_stoi': float(np.mean(stoi_scores)),
            'average_si_sdr': float(np.mean(si_sdr_scores)),
            'processing_time': 3600,  # 1 hour
            'success_rate': 0.95
        },
        'distributions': {
            'pesq': pesq_scores,
            'stoi': stoi_scores,
            'si_sdr': si_sdr_scores
        },
        'trends': {
            'dates': dates,
            'pesq_trend': pesq_trend,
            'stoi_trend': stoi_trend,
            'volume': volume_trend
        },
        'comparisons': {
            'methods': ['Baseline', 'Enhanced', 'Ultra Enhanced'],
            'pesq_scores': [2.5, 3.2, 3.8],
            'stoi_scores': [0.75, 0.87, 0.92],
            'processing_times': [10, 25, 60]
        },
        'issues': {
            'low_quality': 50,
            'processing_failures': 25,
            'timeout_errors': 10
        }
    }


def main():
    """Main demonstration function"""
    print("Quality Report Generator Demo")
    print("=" * 50)
    
    # Generate sample metrics
    print("\n1. Generating sample metrics data...")
    metrics_data = generate_sample_metrics()
    print(f"   - Total samples: {metrics_data['summary']['total_samples']:,}")
    print(f"   - Average PESQ: {metrics_data['summary']['average_pesq']:.2f}")
    print(f"   - Average STOI: {metrics_data['summary']['average_stoi']:.3f}")
    
    # Create report generator
    print("\n2. Creating report generator...")
    generator = QualityReportGenerator(template='standard')
    
    # Generate insights
    print("\n3. Generating insights...")
    insights = generator.generate_insights(metrics_data)
    print(f"   - Found {len(insights)} insights")
    for i, insight in enumerate(insights[:3]):  # Show first 3
        print(f"   - [{insight['severity'].upper()}] {insight['message']}")
    
    # Generate PDF report
    print("\n4. Generating PDF report...")
    pdf_path = generator.generate_report(
        metrics_data,
        output_path='demo_quality_report.pdf',
        format='pdf'
    )
    print(f"   - PDF saved to: {pdf_path}")
    
    # Generate HTML report
    print("\n5. Generating HTML report...")
    html_path = generator.generate_report(
        metrics_data,
        output_path='demo_quality_report.html',
        format='html'
    )
    print(f"   - HTML saved to: {html_path}")
    
    # Generate JSON report
    print("\n6. Generating JSON report...")
    json_path = generator.generate_report(
        metrics_data,
        output_path='demo_quality_report.json',
        format='json'
    )
    print(f"   - JSON saved to: {json_path}")
    
    # Test executive summary template
    print("\n7. Testing executive summary template...")
    exec_generator = QualityReportGenerator(template='executive_summary')
    exec_path = exec_generator.generate_report(
        metrics_data,
        output_path='demo_executive_summary.pdf',
        format='pdf'
    )
    print(f"   - Executive summary saved to: {exec_path}")
    
    # Test batch reporting
    print("\n8. Testing batch report generation...")
    batch_data = {}
    for i in range(3):
        batch_metrics = generate_sample_metrics()
        batch_metrics['summary']['average_pesq'] = 3.0 + i * 0.2
        batch_data[f'dataset_{i+1}'] = batch_metrics
    
    report_paths = generator.generate_batch_reports(
        batch_data,
        output_dir='batch_reports',
        format='html'
    )
    print(f"   - Generated {len(report_paths)} batch reports")
    
    # Test accessibility features
    print("\n9. Testing accessibility features...")
    generator.enable_accessibility_features()
    accessible_path = generator.generate_report(
        metrics_data,
        output_path='demo_accessible_report.html',
        format='html'
    )
    print(f"   - Accessible HTML saved to: {accessible_path}")
    
    # Show metadata
    print("\n10. Report metadata:")
    metadata = generator.get_report_metadata()
    print(f"    - Generation time: {metadata['generation_time']:.2f}s")
    print(f"    - Memory usage: {metadata['memory_usage_mb']:.1f} MB")
    print(f"    - Sections generated: {metadata['sections_generated']}")
    
    print("\n✅ Demo completed successfully!")
    print(f"\nGenerated reports:")
    print(f"  - demo_quality_report.pdf")
    print(f"  - demo_quality_report.html")
    print(f"  - demo_quality_report.json")
    print(f"  - demo_executive_summary.pdf")
    print(f"  - demo_accessible_report.html")
    print(f"  - batch_reports/ (3 HTML files)")


if __name__ == "__main__":
    main()