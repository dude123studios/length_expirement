#!/usr/bin/env python3
"""
Test script for the subthought length experiment.
This script runs a small test to validate the implementation.
"""

import os
import sys
import argparse
from experiment_subthought_length import run_experiment, analyze_results, create_visualizations

def test_experiment():
    """Run a small test of the experiment with minimal examples."""
    
    # Test parameters
    test_args = {
        'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct',  # Adjust as needed
        'benchmark_name': 'math',
        'activations_dir': 'activations',  # Adjust path as needed
        'num_examples': 5,  # Small test
        'high_temperature': 1.2,
        'max_new_tokens': 64,  # Smaller for faster testing
        'output_dir': 'test_results'
    }
    
    print("Running test experiment...")
    print(f"Test parameters: {test_args}")
    
    try:
        # Check if activations directory exists
        if not os.path.exists(test_args['activations_dir']):
            print(f"Error: Activations directory '{test_args['activations_dir']}' not found.")
            print("Please ensure you have generated activations first using generate_activations.py")
            return False
        
        # Run experiment
        results = run_experiment(**test_args)
        
        if not results:
            print("No results generated. Check your activations directory and model path.")
            return False
        
        # Analyze results
        stats = analyze_results(results, test_args['output_dir'])
        
        # Create visualizations
        create_visualizations(results, test_args['output_dir'])
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the subthought length experiment")
    parser.add_argument('--model_name_or_path', help="Path to the model (optional)")
    parser.add_argument('--activations_dir', help="Directory containing stored activations (optional)")
    parser.add_argument('--benchmark_name', default='math', help="Benchmark dataset name")
    
    args = parser.parse_args()
    
    # Override test parameters if provided
    if args.model_name_or_path:
        test_args['model_name_or_path'] = args.model_name_or_path
    if args.activations_dir:
        test_args['activations_dir'] = args.activations_dir
    if args.benchmark_name:
        test_args['benchmark_name'] = args.benchmark_name
    
    success = test_experiment()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
