#!/usr/bin/env python3
"""
Experiment: Subthought Length under High vs Low Temperature Sampling

This script implements the experiment plan to compare the length of reasoning 
subthoughts under high-temperature sampling vs. greedy/low-temperature sampling,
starting from the same partial reasoning trace up to a decision token.

Key components:
1. Load stored reasoning traces and identify decision token positions
2. Generate continuations with both greedy and high-temperature sampling
3. Measure subthought length until next decision token
4. Analyze and visualize results
"""

import argparse
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from tqdm import tqdm
import json
import os
from typing import List, Dict, Tuple, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.data_loader import load_benchmark, parse_question, get_prompt
from utils.activations_loader import load_activations_idx_dict
from analysis.token_analysis import SPECIAL_TOKENS

def _single_token_ids_from_strings(tokens_as_text):
    """Map each token string to an id iff it encodes as exactly one token."""
    mapping = {}
    for s in tokens_as_text:
        ids = tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if len(ids) == 1:
            mapping[s] = ids.item()
    return mapping

def find_decision_token_positions(output_ids: List[int], special_token_to_id: Dict[str, int]) -> List[int]:
    """
    Find positions of decision tokens in the output sequence.
    
    Args:
        output_ids: List of token IDs from the stored reasoning trace
        special_token_to_id: Mapping from special token strings to IDs
        
    Returns:
        List of indices where decision tokens appear
    """
    decision_positions = []
    special_token_ids = set(special_token_to_id.values())
    
    for i, token_id in enumerate(output_ids):
        if token_id in special_token_ids:
            decision_positions.append(i)
    
    return decision_positions

def continue_until_decision(
    example_idx: int, 
    continuation_index: int, 
    temperature: float,
    max_new_tokens: int = 128,
    model=None,
    tokenizer=None,
    examples=None,
    activations_dir=None,
    benchmark_name=None,
    special_token_to_id=None
) -> Tuple[int, str, List[int]]:
    """
    Continue generation from a specific index until a decision token is reached.
    
    Args:
        example_idx: Index of the example in the benchmark
        continuation_index: Token index to start continuation from
        temperature: Sampling temperature (0.0 for greedy, >0 for sampling)
        max_new_tokens: Maximum tokens to generate before stopping
        model, tokenizer, examples, activations_dir, benchmark_name: Required for generation
        special_token_to_id: Mapping of special tokens to IDs
        
    Returns:
        Tuple of (subthought_length, generated_text, generated_token_ids)
    """
    # 1) Load stored example tokens/text
    data = load_activations_idx_dict(activations_dir, example_idx)
    stored_ids = data["output_token_ids"]
    
    # Normalize to 1D Long tensor on the right device
    if isinstance(stored_ids, torch.Tensor):
        if stored_ids.dim() == 2:
            stored_ids = stored_ids[0]
        stored_ids = stored_ids.to(device=model.device, dtype=torch.long)
    else:
        stored_ids = torch.tensor(stored_ids, dtype=torch.long, device=model.device)
    
    # Bounds-check & slice the stored assistant tokens
    T = stored_ids.numel()
    k = int(max(0, min(int(continuation_index), T)))
    stored_prefix_ids = stored_ids[:k]
    if stored_prefix_ids.numel() == 0:
        stored_prefix_ids = torch.empty((0,), dtype=torch.long, device=model.device)
    
    # 2) Rebuild the original chat base (system+user) exactly like the tracing pipeline
    example = examples[example_idx]
    question = parse_question(example)
    messages = get_prompt(question, "qwen-instruct", benchmark_name)
    
    base_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", skip_special_tokens=False
    )[0].to(device=model.device, dtype=torch.long)
    
    # 3) Build the full input_ids: base + stored prefix up to k
    input_ids = torch.cat([base_ids, stored_prefix_ids], dim=0)
    input_ids = input_ids.unsqueeze(0).to(dtype=torch.long)
    
    # 4) Generate with specified temperature
    do_sample = temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 5) Extract newly generated tokens
    seq = out.sequences[0]
    new_len = seq.size(0) - input_ids.size(1)
    gen_ids = seq[-new_len:].tolist() if new_len > 0 else []
    
    # 6) Find first decision token in generated sequence
    special_token_ids = set(special_token_to_id.values())
    subthought_length = max_new_tokens  # Default if no decision token found
    
    for i, token_id in enumerate(gen_ids):
        if token_id in special_token_ids:
            subthought_length = i + 1  # +1 because we count the decision token
            break
    
    # 7) Decode the generated text
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    return subthought_length, generated_text, gen_ids

def run_experiment(
    model_name_or_path: str,
    benchmark_name: str,
    activations_dir: str,
    num_examples: int = 100,
    high_temperature: float = 1.2,
    max_new_tokens: int = 128,
    output_dir: str = "experiment_results"
) -> List[Dict]:
    """
    Run the main experiment comparing subthought lengths under different temperatures.
    
    Args:
        model_name_or_path: Path to the model
        benchmark_name: Name of the benchmark dataset
        activations_dir: Directory containing stored activations
        num_examples: Number of examples to process
        high_temperature: Temperature for high-temperature sampling
        max_new_tokens: Maximum tokens to generate
        output_dir: Directory to save results
        
    Returns:
        List of experiment results
    """
    # Setup
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    
    # Load data
    examples = load_benchmark(benchmark_name)
    special_token_to_id = _single_token_ids_from_strings(SPECIAL_TOKENS)
    
    # Limit examples if specified
    if num_examples > 0:
        examples = examples[:num_examples]
    
    results = []
    
    print(f"Running experiment on {len(examples)} examples...")
    print(f"High temperature: {high_temperature}")
    print(f"Max new tokens: {max_new_tokens}")
    
    for example_idx in tqdm(range(len(examples)), desc="Processing examples"):
        try:
            # Load stored activations for this example
            data = load_activations_idx_dict(activations_dir, example_idx)
            output_ids = data["output_token_ids"]
            
            # Convert to list if tensor
            if isinstance(output_ids, torch.Tensor):
                if output_ids.dim() == 2:
                    output_ids = output_ids[0]
                output_ids = output_ids.tolist()
            
            # Find decision token positions
            decision_positions = find_decision_token_positions(output_ids, special_token_to_id)
            
            if not decision_positions:
                print(f"No decision tokens found in example {example_idx}, skipping...")
                continue
            
            # For each decision token position, run the experiment
            for decision_pos in decision_positions:
                # Skip if we're too close to the end
                if decision_pos >= len(output_ids) - 1:
                    continue
                
                continuation_index = decision_pos  # Start just before the decision token
                
                # Run greedy generation
                greedy_length, greedy_text, greedy_tokens = continue_until_decision(
                    example_idx=example_idx,
                    continuation_index=continuation_index,
                    temperature=0.0,  # Greedy
                    max_new_tokens=max_new_tokens,
                    model=model,
                    tokenizer=tokenizer,
                    examples=examples,
                    activations_dir=activations_dir,
                    benchmark_name=benchmark_name,
                    special_token_to_id=special_token_to_id
                )
                
                # Run high-temperature generation
                high_temp_length, high_temp_text, high_temp_tokens = continue_until_decision(
                    example_idx=example_idx,
                    continuation_index=continuation_index,
                    temperature=high_temperature,
                    max_new_tokens=max_new_tokens,
                    model=model,
                    tokenizer=tokenizer,
                    examples=examples,
                    activations_dir=activations_dir,
                    benchmark_name=benchmark_name,
                    special_token_to_id=special_token_to_id
                )
                
                # Store results
                result = {
                    "example_idx": example_idx,
                    "decision_position": decision_pos,
                    "continuation_index": continuation_index,
                    "greedy_length": greedy_length,
                    "high_temp_length": high_temp_length,
                    "length_difference": high_temp_length - greedy_length,
                    "greedy_text": greedy_text,
                    "high_temp_text": high_temp_text,
                    "greedy_tokens": greedy_tokens,
                    "high_temp_tokens": high_temp_tokens,
                    "temperature": high_temperature
                }
                results.append(result)
                
        except Exception as e:
            print(f"Error processing example {example_idx}: {e}")
            continue
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"subthought_length_results_{benchmark_name}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Experiment completed. Results saved to {results_file}")
    print(f"Total comparisons: {len(results)}")
    
    return results

def analyze_results(results: List[Dict], output_dir: str) -> Dict:
    """
    Analyze the experiment results and generate statistics.
    
    Args:
        results: List of experiment results
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary containing analysis results
    """
    if not results:
        print("No results to analyze!")
        return {}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Basic statistics
    greedy_lengths = df['greedy_length'].values
    high_temp_lengths = df['high_temp_length'].values
    length_differences = df['length_difference'].values
    
    stats_dict = {
        "n_comparisons": len(results),
        "greedy_mean": np.mean(greedy_lengths),
        "greedy_std": np.std(greedy_lengths),
        "greedy_median": np.median(greedy_lengths),
        "high_temp_mean": np.mean(high_temp_lengths),
        "high_temp_std": np.std(high_temp_lengths),
        "high_temp_median": np.median(high_temp_lengths),
        "difference_mean": np.mean(length_differences),
        "difference_std": np.std(length_differences),
        "difference_median": np.median(length_differences)
    }
    
    # Statistical tests
    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(high_temp_lengths, greedy_lengths)
    stats_dict["paired_t_test"] = {"statistic": t_stat, "p_value": t_pvalue}
    
    # Wilcoxon signed-rank test (non-parametric)
    wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(high_temp_lengths, greedy_lengths)
    stats_dict["wilcoxon_test"] = {"statistic": wilcoxon_stat, "p_value": wilcoxon_pvalue}
    
    # Effect size (Cohen's d for paired samples)
    pooled_std = np.sqrt((np.var(greedy_lengths) + np.var(high_temp_lengths)) / 2)
    cohens_d = np.mean(length_differences) / pooled_std
    stats_dict["cohens_d"] = cohens_d
    
    # Save statistics
    stats_file = os.path.join(output_dir, "analysis_statistics.json")
    with open(stats_file, "w") as f:
        json.dump(stats_dict, f, indent=2)
    
    print("Analysis Results:")
    print(f"Number of comparisons: {stats_dict['n_comparisons']}")
    print(f"Greedy mean length: {stats_dict['greedy_mean']:.2f} ± {stats_dict['greedy_std']:.2f}")
    print(f"High-temp mean length: {stats_dict['high_temp_mean']:.2f} ± {stats_dict['high_temp_std']:.2f}")
    print(f"Mean difference: {stats_dict['difference_mean']:.2f} ± {stats_dict['difference_std']:.2f}")
    print(f"Paired t-test p-value: {stats_dict['paired_t_test']['p_value']:.6f}")
    print(f"Wilcoxon test p-value: {stats_dict['wilcoxon_test']['p_value']:.6f}")
    print(f"Effect size (Cohen's d): {stats_dict['cohens_d']:.3f}")
    
    return stats_dict

def create_visualizations(results: List[Dict], output_dir: str):
    """
    Create visualizations for the experiment results.
    
    Args:
        results: List of experiment results
        output_dir: Directory to save visualizations
    """
    if not results:
        print("No results to visualize!")
        return
    
    df = pd.DataFrame(results)
    
    # 1. Box plot comparing distributions
    fig_box = go.Figure()
    
    fig_box.add_trace(go.Box(
        y=df['greedy_length'],
        name='Greedy',
        boxpoints='outliers',
        marker_color='lightblue'
    ))
    
    fig_box.add_trace(go.Box(
        y=df['high_temp_length'],
        name='High Temperature',
        boxpoints='outliers',
        marker_color='lightcoral'
    ))
    
    fig_box.update_layout(
        title='Distribution of Subthought Lengths: Greedy vs High Temperature',
        yaxis_title='Subthought Length (tokens)',
        xaxis_title='Sampling Strategy',
        showlegend=True
    )
    
    box_file = os.path.join(output_dir, "subthought_length_boxplot.html")
    fig_box.write_html(box_file)
    
    # 2. Scatter plot: Greedy vs High Temperature
    fig_scatter = go.Figure()
    
    fig_scatter.add_trace(go.Scatter(
        x=df['greedy_length'],
        y=df['high_temp_length'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['length_difference'],
            colorscale='RdBu',
            colorbar=dict(title="Length Difference"),
            opacity=0.7
        ),
        text=[f"Example {row['example_idx']}, Decision {row['decision_position']}" 
              for _, row in df.iterrows()],
        hovertemplate="Greedy: %{x}<br>High-temp: %{y}<br>%{text}<extra></extra>"
    ))
    
    # Add diagonal line (y=x)
    max_val = max(df['greedy_length'].max(), df['high_temp_length'].max())
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Equal Lengths'
    ))
    
    fig_scatter.update_layout(
        title='Subthought Lengths: Greedy vs High Temperature',
        xaxis_title='Greedy Length (tokens)',
        yaxis_title='High Temperature Length (tokens)',
        showlegend=True
    )
    
    scatter_file = os.path.join(output_dir, "subthought_length_scatter.html")
    fig_scatter.write_html(scatter_file)
    
    # 3. Histogram of differences
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Histogram(
        x=df['length_difference'],
        nbinsx=30,
        marker_color='lightgreen',
        opacity=0.7
    ))
    
    # Add vertical line at 0
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red", 
                       annotation_text="No Difference")
    
    fig_hist.update_layout(
        title='Distribution of Length Differences (High-temp - Greedy)',
        xaxis_title='Length Difference (tokens)',
        yaxis_title='Frequency',
        showlegend=False
    )
    
    hist_file = os.path.join(output_dir, "length_difference_histogram.html")
    fig_hist.write_html(hist_file)
    
    # 4. Combined subplot
    fig_combined = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Length Distributions', 'Greedy vs High-temp', 
                       'Length Differences', 'Summary Statistics'),
        specs=[[{"type": "box"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "bar"}]]
    )
    
    # Box plots
    fig_combined.add_trace(go.Box(y=df['greedy_length'], name='Greedy'), row=1, col=1)
    fig_combined.add_trace(go.Box(y=df['high_temp_length'], name='High-temp'), row=1, col=1)
    
    # Scatter plot
    fig_combined.add_trace(go.Scatter(
        x=df['greedy_length'], y=df['high_temp_length'], mode='markers',
        name='Comparisons', showlegend=False
    ), row=1, col=2)
    
    # Histogram
    fig_combined.add_trace(go.Histogram(
        x=df['length_difference'], name='Differences', showlegend=False
    ), row=2, col=1)
    
    # Summary bar chart
    summary_data = {
        'Greedy Mean': [df['greedy_length'].mean()],
        'High-temp Mean': [df['high_temp_length'].mean()],
        'Difference Mean': [df['length_difference'].mean()]
    }
    
    for name, values in summary_data.items():
        fig_combined.add_trace(go.Bar(
            x=[name], y=values, name=name, showlegend=False
        ), row=2, col=2)
    
    fig_combined.update_layout(
        title='Subthought Length Experiment: Complete Analysis',
        height=800,
        showlegend=True
    )
    
    combined_file = os.path.join(output_dir, "complete_analysis.html")
    fig_combined.write_html(combined_file)
    
    print(f"Visualizations saved to {output_dir}/")
    print(f"- Box plot: {box_file}")
    print(f"- Scatter plot: {scatter_file}")
    print(f"- Histogram: {hist_file}")
    print(f"- Combined analysis: {combined_file}")

def main():
    parser = argparse.ArgumentParser(description="Subthought Length Temperature Experiment")
    parser.add_argument('--model_name_or_path', required=True, help="Path to the model")
    parser.add_argument('--benchmark_name', type=str, default="math", help="Benchmark dataset name")
    parser.add_argument('--activations_dir', required=True, help="Directory containing stored activations")
    parser.add_argument('--num_examples', type=int, default=100, help="Number of examples to process")
    parser.add_argument('--high_temperature', type=float, default=1.2, help="High temperature for sampling")
    parser.add_argument('--max_new_tokens', type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument('--output_dir', type=str, default="experiment_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Run the experiment
    results = run_experiment(
        model_name_or_path=args.model_name_or_path,
        benchmark_name=args.benchmark_name,
        activations_dir=args.activations_dir,
        num_examples=args.num_examples,
        high_temperature=args.high_temperature,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir
    )
    
    # Analyze results
    stats = analyze_results(results, args.output_dir)
    
    # Create visualizations
    create_visualizations(results, args.output_dir)
    
    print("\nExperiment completed successfully!")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
