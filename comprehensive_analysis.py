#!/usr/bin/env python3
"""
Comprehensive analysis script for scale experiment results
Generates detailed statistics, IQR analysis, and publication-ready summaries
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import argparse

def load_results(results_file):
    """Load experiment results from JSON file"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def calculate_comprehensive_statistics(results_data):
    """Calculate comprehensive statistics including IQR and effect sizes"""
    
    # Extract data
    results = results_data['results']
    regular_lengths = [r['regular_position'] for r in results]
    high_temp_lengths = [r['high_temp_position'] for r in results]
    differences = [r['high_temp_position'] for r in results]  # This should be regular - high
    
    # Fix differences calculation
    differences = [r['regular_position'] - r['high_temp_position'] for r in results]
    
    # Basic statistics
    stats_dict = {
        'sample_size': len(results),
        'regular': {
            'mean': np.mean(regular_lengths),
            'std': np.std(regular_lengths),
            'median': np.median(regular_lengths),
            'q25': np.percentile(regular_lengths, 25),
            'q75': np.percentile(regular_lengths, 75),
            'iqr': np.percentile(regular_lengths, 75) - np.percentile(regular_lengths, 25),
            'min': np.min(regular_lengths),
            'max': np.max(regular_lengths),
            'range': np.max(regular_lengths) - np.min(regular_lengths)
        },
        'high_temp': {
            'mean': np.mean(high_temp_lengths),
            'std': np.std(high_temp_lengths),
            'median': np.median(high_temp_lengths),
            'q25': np.percentile(high_temp_lengths, 25),
            'q75': np.percentile(high_temp_lengths, 75),
            'iqr': np.percentile(high_temp_lengths, 75) - np.percentile(high_temp_lengths, 25),
            'min': np.min(high_temp_lengths),
            'max': np.max(high_temp_lengths),
            'range': np.max(high_temp_lengths) - np.min(high_temp_lengths)
        },
        'differences': {
            'mean': np.mean(differences),
            'std': np.std(differences),
            'median': np.median(differences),
            'q25': np.percentile(differences, 25),
            'q75': np.percentile(differences, 75),
            'iqr': np.percentile(differences, 75) - np.percentile(differences, 25),
            'min': np.min(differences),
            'max': np.max(differences),
            'range': np.max(differences) - np.min(differences)
        }
    }
    
    # Statistical tests
    t_stat, p_value = stats.ttest_rel(regular_lengths, high_temp_lengths)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(regular_lengths, high_temp_lengths)
    
    # Effect sizes
    cohens_d = np.mean(differences) / np.std(differences)
    glass_delta = np.mean(differences) / np.std(regular_lengths)
    
    # Correlation
    correlation, corr_p = stats.pearsonr(regular_lengths, high_temp_lengths)
    
    stats_dict.update({
        'statistical_tests': {
            'paired_t_test': {
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'wilcoxon_signed_rank': {
                'statistic': wilcoxon_stat,
                'p_value': wilcoxon_p,
                'significant': wilcoxon_p < 0.05
            }
        },
        'effect_sizes': {
            'cohens_d': cohens_d,
            'glass_delta': glass_delta,
            'interpretation': interpret_effect_size(abs(cohens_d))
        },
        'correlation': {
            'pearson_r': correlation,
            'p_value': corr_p,
            'significant': corr_p < 0.05
        }
    })
    
    return stats_dict, regular_lengths, high_temp_lengths, differences

def interpret_effect_size(d):
    """Interpret Cohen's d effect size"""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def create_comprehensive_plots(regular_lengths, high_temp_lengths, differences, output_dir, args_data):
    """Create comprehensive visualization plots"""
    
    regular_temp = args_data.get('low_temp', 0.7)
    high_temp = args_data.get('high_temp', 3.0)
    
    # Create comprehensive subplot figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Length Distributions', 'Length Differences',
            'Box Plot Comparison', 'Q-Q Plot (Normality)',
            'Scatter Plot (Paired)', 'Cumulative Distributions'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Length distributions
    fig.add_trace(go.Histogram(
        x=regular_lengths, name=f'Regular (T={regular_temp})',
        opacity=0.7, nbinsx=20, marker_color='#2E86AB'
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=high_temp_lengths, name=f'High (T={high_temp})',
        opacity=0.7, nbinsx=20, marker_color='#A23B72'
    ), row=1, col=1)
    
    # 2. Length differences
    fig.add_trace(go.Histogram(
        x=differences, name='Differences (Regular - High)',
        nbinsx=20, marker_color='#F18F01', opacity=0.8
    ), row=1, col=2)
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
    
    # 3. Box plot comparison
    fig.add_trace(go.Box(
        y=regular_lengths, name=f'Regular (T={regular_temp})',
        marker_color='#2E86AB'
    ), row=2, col=1)
    
    fig.add_trace(go.Box(
        y=high_temp_lengths, name=f'High (T={high_temp})',
        marker_color='#A23B72'
    ), row=2, col=1)
    
    # 4. Q-Q plot for differences (normality check)
    from scipy.stats import probplot
    qq_data = probplot(differences, dist="norm")
    fig.add_trace(go.Scatter(
        x=qq_data[0][0], y=qq_data[0][1],
        mode='markers', name='Q-Q Plot',
        marker_color='#28A745'
    ), row=2, col=2)
    
    # Add theoretical line
    fig.add_trace(go.Scatter(
        x=qq_data[0][0], y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
        mode='lines', name='Theoretical',
        line_color='red', line_dash='dash'
    ), row=2, col=2)
    
    # 5. Scatter plot (paired)
    fig.add_trace(go.Scatter(
        x=regular_lengths, y=high_temp_lengths,
        mode='markers', name='Paired Data',
        marker_color='#6C757D', marker_size=8
    ), row=3, col=1)
    
    # Add diagonal line
    max_val = max(max(regular_lengths), max(high_temp_lengths))
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines', name='Equal',
        line_color='red', line_dash='dash'
    ), row=3, col=1)
    
    # 6. Cumulative distributions
    regular_sorted = np.sort(regular_lengths)
    high_temp_sorted = np.sort(high_temp_lengths)
    regular_cumsum = np.arange(1, len(regular_sorted) + 1) / len(regular_sorted)
    high_temp_cumsum = np.arange(1, len(high_temp_sorted) + 1) / len(high_temp_sorted)
    
    fig.add_trace(go.Scatter(
        x=regular_sorted, y=regular_cumsum,
        mode='lines', name=f'Regular (T={regular_temp})',
        line_color='#2E86AB'
    ), row=3, col=2)
    
    fig.add_trace(go.Scatter(
        x=high_temp_sorted, y=high_temp_cumsum,
        mode='lines', name=f'High (T={high_temp})',
        line_color='#A23B72'
    ), row=3, col=2)
    
    # Update layout
    fig.update_layout(
        title='Comprehensive Subthought Length Analysis',
        height=1200,
        width=1200,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Tokens to Decision Token", row=1, col=1)
    fig.update_xaxes(title_text="Length Difference (tokens)", row=1, col=2)
    fig.update_xaxes(title_text="Method", row=2, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_xaxes(title_text=f"Regular (T={regular_temp})", row=3, col=1)
    fig.update_xaxes(title_text="Tokens to Decision Token", row=3, col=2)
    
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Tokens to Decision Token", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
    fig.update_yaxes(title_text=f"High (T={high_temp})", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative Probability", row=3, col=2)
    
    # Save plot
    comprehensive_file = os.path.join(output_dir, 'comprehensive_analysis.html')
    fig.write_html(comprehensive_file)
    print(f"📊 Comprehensive analysis plot saved to {comprehensive_file}")
    
    return fig

def create_iqr_analysis_plot(regular_lengths, high_temp_lengths, differences, output_dir, args_data):
    """Create detailed IQR analysis plot"""
    
    regular_temp = args_data.get('low_temp', 0.7)
    high_temp = args_data.get('high_temp', 3.0)
    
    # Calculate IQR statistics
    regular_q25, regular_q75 = np.percentile(regular_lengths, [25, 75])
    high_temp_q25, high_temp_q75 = np.percentile(high_temp_lengths, [25, 75])
    diff_q25, diff_q75 = np.percentile(differences, [25, 75])
    
    # Create IQR comparison plot
    fig = go.Figure()
    
    # Add box plots with IQR annotations
    fig.add_trace(go.Box(
        y=regular_lengths, name=f'Regular (T={regular_temp})',
        boxpoints='outliers', marker_color='#2E86AB',
        hovertemplate='<b>Regular Temperature</b><br>' +
                     f'Q25: {regular_q25:.1f}<br>' +
                     f'Q75: {regular_q75:.1f}<br>' +
                     f'IQR: {regular_q75 - regular_q25:.1f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.add_trace(go.Box(
        y=high_temp_lengths, name=f'High (T={high_temp})',
        boxpoints='outliers', marker_color='#A23B72',
        hovertemplate='<b>High Temperature</b><br>' +
                     f'Q25: {high_temp_q25:.1f}<br>' +
                     f'Q75: {high_temp_q75:.1f}<br>' +
                     f'IQR: {high_temp_q75 - high_temp_q25:.1f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title='IQR Analysis: Regular vs High Temperature',
        yaxis_title='Tokens to Decision Token',
        xaxis_title='Temperature Setting',
        height=600,
        width=800
    )
    
    # Save IQR plot
    iqr_file = os.path.join(output_dir, 'iqr_analysis.html')
    fig.write_html(iqr_file)
    print(f"📊 IQR analysis plot saved to {iqr_file}")
    
    return fig

def generate_publication_summary(stats_dict, output_dir):
    """Generate publication-ready summary"""
    
    summary = f"""
# Subthought Length Experiment Results

## Experimental Design
- **Model**: DeepSeek-R1-Distill-Qwen-7B
- **Sample Size**: {stats_dict['sample_size']} examples
- **Regular Temperature**: 0.0 (greedy)
- **High Temperature**: 3.0
- **Stopping Condition**: Highest probability decision token
- **Max Tokens**: 128

## Descriptive Statistics

### Regular Temperature (T=0.0, Greedy)
- **Mean**: {stats_dict['regular']['mean']:.2f} ± {stats_dict['regular']['std']:.2f}
- **Median**: {stats_dict['regular']['median']:.1f}
- **IQR**: {stats_dict['regular']['iqr']:.1f} (Q25: {stats_dict['regular']['q25']:.1f}, Q75: {stats_dict['regular']['q75']:.1f})
- **Range**: {stats_dict['regular']['range']:.1f} ({stats_dict['regular']['min']:.1f} - {stats_dict['regular']['max']:.1f})

### High Temperature (T=3.0)
- **Mean**: {stats_dict['high_temp']['mean']:.2f} ± {stats_dict['high_temp']['std']:.2f}
- **Median**: {stats_dict['high_temp']['median']:.1f}
- **IQR**: {stats_dict['high_temp']['iqr']:.1f} (Q25: {stats_dict['high_temp']['q25']:.1f}, Q75: {stats_dict['high_temp']['q75']:.1f})
- **Range**: {stats_dict['high_temp']['range']:.1f} ({stats_dict['high_temp']['min']:.1f} - {stats_dict['high_temp']['max']:.1f})

### Differences (Regular - High)
- **Mean**: {stats_dict['differences']['mean']:.2f} ± {stats_dict['differences']['std']:.2f}
- **Median**: {stats_dict['differences']['median']:.1f}
- **IQR**: {stats_dict['differences']['iqr']:.1f} (Q25: {stats_dict['differences']['q25']:.1f}, Q75: {stats_dict['differences']['q75']:.1f})
- **Range**: {stats_dict['differences']['range']:.1f} ({stats_dict['differences']['min']:.1f} - {stats_dict['differences']['max']:.1f})

## Statistical Analysis

### Paired t-test
- **t-statistic**: {stats_dict['statistical_tests']['paired_t_test']['statistic']:.3f}
- **p-value**: {stats_dict['statistical_tests']['paired_t_test']['p_value']:.4f}
- **Significant**: {'Yes' if stats_dict['statistical_tests']['paired_t_test']['significant'] else 'No'}

### Wilcoxon Signed-Rank Test
- **W-statistic**: {stats_dict['statistical_tests']['wilcoxon_signed_rank']['statistic']:.3f}
- **p-value**: {stats_dict['statistical_tests']['wilcoxon_signed_rank']['p_value']:.4f}
- **Significant**: {'Yes' if stats_dict['statistical_tests']['wilcoxon_signed_rank']['significant'] else 'No'}

## Effect Sizes
- **Cohen's d**: {stats_dict['effect_sizes']['cohens_d']:.3f} ({stats_dict['effect_sizes']['interpretation']} effect)
- **Glass's Δ**: {stats_dict['effect_sizes']['glass_delta']:.3f}

## Correlation
- **Pearson r**: {stats_dict['correlation']['pearson_r']:.3f}
- **p-value**: {stats_dict['correlation']['p_value']:.4f}
- **Significant**: {'Yes' if stats_dict['correlation']['significant'] else 'No'}

## Interpretation
{'High temperature leads to shorter subthoughts' if stats_dict['differences']['mean'] > 0 else 'High temperature leads to longer subthoughts'} (mean difference: {stats_dict['differences']['mean']:.2f} tokens).

The effect size is {stats_dict['effect_sizes']['interpretation']} (Cohen's d = {stats_dict['effect_sizes']['cohens_d']:.3f}).
"""
    
    # Save summary
    summary_file = os.path.join(output_dir, 'publication_summary.md')
    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"📄 Publication summary saved to {summary_file}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Comprehensive analysis of scale experiment results')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to the JSON results file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis (default: same as results file)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results_file)
    
    print("📊 Running Comprehensive Analysis...")
    print(f"📁 Results file: {args.results_file}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Load results
    results_data = load_results(args.results_file)
    
    # Calculate comprehensive statistics
    stats_dict, regular_lengths, high_temp_lengths, differences = calculate_comprehensive_statistics(results_data)
    
    # Save detailed statistics
    stats_file = os.path.join(args.output_dir, 'comprehensive_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"📊 Comprehensive statistics saved to {stats_file}")
    
    # Create comprehensive plots
    create_comprehensive_plots(regular_lengths, high_temp_lengths, differences, 
                              args.output_dir, results_data['args'])
    
    # Create IQR analysis
    create_iqr_analysis_plot(regular_lengths, high_temp_lengths, differences, 
                            args.output_dir, results_data['args'])
    
    # Generate publication summary
    generate_publication_summary(stats_dict, args.output_dir)
    
    print("✅ Comprehensive analysis completed!")
    print(f"📁 Check {args.output_dir}/ for all analysis files")

if __name__ == "__main__":
    main()
