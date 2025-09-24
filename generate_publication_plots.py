#!/usr/bin/env python3
"""
Standalone script to generate publication-ready plots from existing experiment results
"""

import json
import os
import sys
import argparse
from publication_plots import create_publication_plots

def main():
    parser = argparse.ArgumentParser(description='Generate publication-ready plots from experiment results')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to the JSON results file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as results file)')
    parser.add_argument('--model_name', type=str, default='DeepSeek-R1-Distill-Qwen-7B',
                       help='Model name for plot titles')
    
    args = parser.parse_args()
    
    # Load results
    print(f"📥 Loading results from {args.results_file}")
    try:
        with open(args.results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading results file: {e}")
        return
    
    # Extract results and parameters
    results = data.get('results', [])
    if not results:
        print("❌ No results found in file")
        return
    
    # Get parameters from args or data
    args_data = data.get('args', {})
    regular_temp = args_data.get('low_temp', 0.7)
    high_temp = args_data.get('high_temp', 1.8)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results_file)
    
    print(f"📊 Creating publication plots...")
    print(f"   Results: {len(results)} examples")
    print(f"   Regular temp: {regular_temp}")
    print(f"   High temp: {high_temp}")
    print(f"   Output: {args.output_dir}")
    
    # Create plots
    create_publication_plots(results, args.output_dir, regular_temp, high_temp, args.model_name)
    
    print("✅ Publication plots generated successfully!")
    print(f"📁 Check {args.output_dir}/ for:")
    print("   - main_comparison_publication.pdf")
    print("   - probability_evolution_publication.pdf") 
    print("   - statistical_summary_publication.pdf")
    print("   - example_*_probabilities_publication.pdf")

if __name__ == "__main__":
    main()
