#!/usr/bin/env python3
"""
High-quality publication-ready plotting functions for the subthought length experiment
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import os
from scipy import stats

# Set publication-quality matplotlib settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.2
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['xtick.major.size'] = 5
rcParams['xtick.minor.size'] = 3
rcParams['ytick.major.size'] = 5
rcParams['ytick.minor.size'] = 3
rcParams['legend.frameon'] = False
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1

# Publication color palette
PUBLICATION_COLORS = {
    'regular': '#2E86AB',      # Blue
    'high_temp': '#A23B72',    # Red/Pink
    'neutral': '#6C757D',      # Gray
    'accent': '#F18F01',       # Orange
    'success': '#28A745',      # Green
    'text': '#212529'          # Dark gray
}

def create_publication_plots(results, output_dir, regular_temp, high_temp, model_name="DeepSeek-R1-Distill-Qwen-7B"):
    """Create all publication-ready plots"""
    
    print("📊 Creating publication-ready plots...")
    
    # Extract data
    regular_positions = [r['regular_position'] for r in results]
    high_temp_positions = [r['high_temp_position'] for r in results]
    differences = [h - r for h, r in zip(high_temp_positions, regular_positions)]
    
    # Create main comparison plot
    create_main_comparison_plot(regular_positions, high_temp_positions, regular_temp, high_temp, output_dir, model_name)
    
    # Create probability evolution plot
    create_probability_evolution_plot(results, regular_temp, high_temp, output_dir, model_name)
    
    # Create statistical summary plot
    create_statistical_summary_plot(regular_positions, high_temp_positions, differences, regular_temp, high_temp, output_dir, model_name)
    
    # Create individual example plots
    create_individual_example_plots(results, regular_temp, high_temp, output_dir, model_name)
    
    print("✅ All publication plots created!")

def create_main_comparison_plot(regular_positions, high_temp_positions, regular_temp, high_temp, output_dir, model_name):
    """Create the main comparison plot with box plots and statistical annotations"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Prepare data
    data = pd.DataFrame({
        'Temperature': ['Regular'] * len(regular_positions) + ['High'] * len(high_temp_positions),
        'Position': regular_positions + high_temp_positions,
        'Method': ['Regular'] * len(regular_positions) + ['High'] * len(high_temp_positions)
    })
    
    # Box plot
    sns.boxplot(data=data, x='Temperature', y='Position', ax=ax1, 
                palette=[PUBLICATION_COLORS['regular'], PUBLICATION_COLORS['high_temp']],
                width=0.6, linewidth=1.5)
    
    # Add individual points
    sns.stripplot(data=data, x='Temperature', y='Position', ax=ax1, 
                  color='black', alpha=0.4, size=4, jitter=0.2)
    
    ax1.set_ylabel('Tokens to Decision Token', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Temperature Setting', fontsize=14, fontweight='bold')
    ax1.set_title('A. Subthought Length Comparison', fontsize=16, fontweight='bold', pad=20)
    
    # Add statistical annotations
    regular_mean = np.mean(regular_positions)
    high_temp_mean = np.mean(high_temp_positions)
    t_stat, p_value = stats.ttest_rel(high_temp_positions, regular_positions)
    
    # Add mean lines
    ax1.axhline(y=regular_mean, xmin=0.1, xmax=0.4, color=PUBLICATION_COLORS['regular'], 
                linestyle='--', linewidth=2, alpha=0.8)
    ax1.axhline(y=high_temp_mean, xmin=0.6, xmax=0.9, color=PUBLICATION_COLORS['high_temp'], 
                linestyle='--', linewidth=2, alpha=0.8)
    
    # Add statistical text
    stats_text = f'Mean difference: {high_temp_mean - regular_mean:.1f} tokens\n'
    stats_text += f't-test: t = {t_stat:.2f}, p = {p_value:.3f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Scatter plot
    ax2.scatter(regular_positions, high_temp_positions, 
                c=PUBLICATION_COLORS['accent'], s=60, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add diagonal line
    max_val = max(max(regular_positions), max(high_temp_positions))
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1.5, label='Equal positions')
    
    # Add correlation
    corr, corr_p = stats.pearsonr(regular_positions, high_temp_positions)
    ax2.text(0.05, 0.95, f'r = {corr:.3f}\np = {corr_p:.3f}', 
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel(f'Regular Temperature (T={regular_temp})', fontsize=14, fontweight='bold')
    ax2.set_ylabel(f'High Temperature (T={high_temp})', fontsize=14, fontweight='bold')
    ax2.set_title('B. Paired Comparison', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=10)
    
    # Add grid
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save high-quality version
    plt.savefig(os.path.join(output_dir, 'main_comparison_publication.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'main_comparison_publication.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"📊 Main comparison plot saved to {output_dir}/main_comparison_publication.pdf")

def create_probability_evolution_plot(results, regular_temp, high_temp, output_dir, model_name):
    """Create probability evolution over time plot"""
    
    # Collect probability data
    regular_probs_data = []
    high_temp_probs_data = []
    
    for result in results:
        if 'regular_probs_over_time' in result and 'high_temp_probs_over_time' in result:
            for step_data in result['regular_probs_over_time']:
                regular_probs_data.append({
                    'step': step_data['step'],
                    'max_prob': step_data['max_decision_prob'],
                    'token': step_data['max_decision_token']
                })
            
            for step_data in result['high_temp_probs_over_time']:
                high_temp_probs_data.append({
                    'step': step_data['step'],
                    'max_prob': step_data['max_decision_prob'],
                    'token': step_data['max_decision_token']
                })
    
    if not regular_probs_data or not high_temp_probs_data:
        print("⚠️ No probability data available")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Average probabilities over time
    regular_steps = {}
    for d in regular_probs_data:
        step = d['step']
        if step not in regular_steps:
            regular_steps[step] = []
        regular_steps[step].append(d['max_prob'])
    
    high_temp_steps = {}
    for d in high_temp_probs_data:
        step = d['step']
        if step not in high_temp_steps:
            high_temp_steps[step] = []
        high_temp_steps[step].append(d['max_prob'])
    
    # Plot with error bars
    regular_steps_sorted = sorted(regular_steps.keys())
    regular_means = [np.mean(regular_steps[step]) for step in regular_steps_sorted]
    regular_stds = [np.std(regular_steps[step]) for step in regular_steps_sorted]
    
    high_temp_steps_sorted = sorted(high_temp_steps.keys())
    high_temp_means = [np.mean(high_temp_steps[step]) for step in high_temp_steps_sorted]
    high_temp_stds = [np.std(high_temp_steps[step]) for step in high_temp_steps_sorted]
    
    # Plot lines with error bars
    ax.errorbar(regular_steps_sorted, regular_means, yerr=regular_stds, 
                color=PUBLICATION_COLORS['regular'], linewidth=2.5, marker='o', 
                markersize=6, capsize=3, capthick=1.5, label=f'Regular (T={regular_temp})')
    
    ax.errorbar(high_temp_steps_sorted, high_temp_means, yerr=high_temp_stds, 
                color=PUBLICATION_COLORS['high_temp'], linewidth=2.5, marker='s', 
                markersize=6, capsize=3, capthick=1.5, label=f'High (T={high_temp})')
    
    # Add high-likelihood threshold line
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, linewidth=2, 
               label='High-likelihood threshold (0.3)')
    
    ax.set_xlabel('Generation Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Maximum Decision Token Probability', fontsize=14, fontweight='bold')
    ax.set_title('Decision Token Probability Evolution Over Time', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_evolution_publication.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'probability_evolution_publication.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"📊 Probability evolution plot saved to {output_dir}/probability_evolution_publication.pdf")

def create_statistical_summary_plot(regular_positions, high_temp_positions, differences, regular_temp, high_temp, output_dir, model_name):
    """Create statistical summary plot"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram of differences
    ax1.hist(differences, bins=15, alpha=0.7, color=PUBLICATION_COLORS['neutral'], 
             edgecolor='black', linewidth=1)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.axvline(x=np.mean(differences), color=PUBLICATION_COLORS['accent'], 
                linestyle='-', linewidth=3, alpha=0.8, label=f'Mean: {np.mean(differences):.1f}')
    ax1.set_xlabel('Position Difference (High - Regular)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('A. Distribution of Differences', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot for normality
    stats.probplot(differences, dist="norm", plot=ax2)
    ax2.set_title('B. Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Box plot of individual methods
    data = [regular_positions, high_temp_positions]
    bp = ax3.boxplot(data, labels=[f'Regular\n(T={regular_temp})', f'High\n(T={high_temp})'], 
                     patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(PUBLICATION_COLORS['regular'])
    bp['boxes'][1].set_facecolor(PUBLICATION_COLORS['high_temp'])
    ax3.set_ylabel('Tokens to Decision Token', fontsize=12, fontweight='bold')
    ax3.set_title('C. Method Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Statistical summary
    ax4.axis('off')
    stats_text = f"""
Statistical Summary

Sample size: {len(regular_positions)}
Regular temperature (T={regular_temp}):
  Mean: {np.mean(regular_positions):.2f} ± {np.std(regular_positions):.2f}
  Median: {np.median(regular_positions):.2f}
  Range: {np.min(regular_positions)} - {np.max(regular_positions)}

High temperature (T={high_temp}):
  Mean: {np.mean(high_temp_positions):.2f} ± {np.std(high_temp_positions):.2f}
  Median: {np.median(high_temp_positions):.2f}
  Range: {np.min(high_temp_positions)} - {np.max(high_temp_positions)}

Difference (High - Regular):
  Mean: {np.mean(differences):.2f} ± {np.std(differences):.2f}
  Median: {np.median(differences):.2f}

Statistical Tests:
  Paired t-test: t = {stats.ttest_rel(high_temp_positions, regular_positions)[0]:.3f}
  p-value: {stats.ttest_rel(high_temp_positions, regular_positions)[1]:.3f}
  Effect size (Cohen's d): {np.mean(differences) / np.std(differences):.3f}
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistical_summary_publication.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'statistical_summary_publication.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"📊 Statistical summary plot saved to {output_dir}/statistical_summary_publication.pdf")

def create_individual_example_plots(results, regular_temp, high_temp, output_dir, model_name):
    """Create plots for individual examples"""
    
    # Select a few interesting examples
    interesting_examples = results[:min(3, len(results))]  # First 3 examples
    
    for i, result in enumerate(interesting_examples):
        if 'regular_probs_over_time' not in result or 'high_temp_probs_over_time' not in result:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Regular temperature probabilities
        regular_steps = [d['step'] for d in result['regular_probs_over_time']]
        regular_probs = [d['max_decision_prob'] for d in result['regular_probs_over_time']]
        regular_tokens = [d['max_decision_token'] for d in result['regular_probs_over_time']]
        
        ax1.plot(regular_steps, regular_probs, color=PUBLICATION_COLORS['regular'], 
                 linewidth=2, marker='o', markersize=4)
        ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.set_xlabel('Generation Step', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Max Decision Token Probability', fontsize=12, fontweight='bold')
        ax1.set_title(f'Regular Temperature (T={regular_temp})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # High temperature probabilities
        high_temp_steps = [d['step'] for d in result['high_temp_probs_over_time']]
        high_temp_probs = [d['max_decision_prob'] for d in result['high_temp_probs_over_time']]
        high_temp_tokens = [d['max_decision_token'] for d in result['high_temp_probs_over_time']]
        
        ax2.plot(high_temp_steps, high_temp_probs, color=PUBLICATION_COLORS['high_temp'], 
                 linewidth=2, marker='s', markersize=4)
        ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax2.set_xlabel('Generation Step', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Max Decision Token Probability', fontsize=12, fontweight='bold')
        ax2.set_title(f'High Temperature (T={high_temp})', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add stopping point annotations
        if result['regular_position'] < 128:  # Didn't hit max
            ax1.axvline(x=result['regular_position']-1, color='green', linestyle=':', 
                       linewidth=2, alpha=0.8, label=f'Stopped at step {result["regular_position"]}')
            ax1.legend()
        
        if result['high_temp_position'] < 128:  # Didn't hit max
            ax2.axvline(x=result['high_temp_position']-1, color='green', linestyle=':', 
                       linewidth=2, alpha=0.8, label=f'Stopped at step {result["high_temp_position"]}')
            ax2.legend()
        
        plt.suptitle(f'Example {i+1}: {result["file"]}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'example_{i+1}_probabilities_publication.pdf'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(output_dir, f'example_{i+1}_probabilities_publication.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"📊 Individual example plots saved to {output_dir}/example_*_probabilities_publication.pdf")

if __name__ == "__main__":
    # Example usage
    print("Publication plotting module loaded. Use create_publication_plots() function.")
