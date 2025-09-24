#!/usr/bin/env python3
"""
Experiment: Subthought Length under High vs Low Temperature Sampling
(Text-only version - works with directories containing only text output files)

This version can work with directories that only contain text output files,
generating the necessary token data on-the-fly.
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
from analysis.token_analysis import SPECIAL_TOKENS

def _single_token_ids_from_strings(tokens_as_text, tokenizer):
    """Map each token string to an id iff it encodes as exactly one token."""
    mapping = {}
    for s in tokens_as_text:
        ids = tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if len(ids) == 1:
            mapping[s] = ids.item()
    return mapping

def load_text_outputs(text_dir: str) -> List[Dict]:
    """
    Load text outputs from a directory containing example_*_output.txt files.
    
    Args:
        text_dir: Directory containing text output files
        
    Returns:
        List of dictionaries with example data
    """
    text_files = sorted([f for f in os.listdir(text_dir) if f.startswith("example_") and f.endswith("_output.txt")])
    
    if not text_files:
        raise ValueError(f"No example_*_output.txt files found in {text_dir}")
    
    examples = []
    for text_file in text_files:
        example_idx = int(text_file.replace("example_", "").replace("_output.txt", ""))
        text_path = os.path.join(text_dir, text_file)
        
        with open(text_path, 'r', encoding='utf-8') as f:
            output_text = f.read().strip()
        
        examples.append({
            "example_idx": example_idx,
            "output_text": output_text
        })
    
    return examples

def find_decision_token_positions_in_text(text: str, tokenizer, special_token_to_id: Dict[str, int]) -> List[int]:
    """
    Find positions of decision tokens in the text by tokenizing and checking for special tokens.
    
    Args:
        text: The text to analyze
        tokenizer: Tokenizer to use
        special_token_to_id: Mapping from special token strings to IDs
        
    Returns:
        List of token indices where decision tokens appear
    """
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
    token_strs = tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=False)
    
    decision_positions = []
    special_token_ids = set(special_token_to_id.values())
    
    for i, token_id in enumerate(tokens.tolist()):
        if token_id in special_token_ids:
            decision_positions.append(i)
    
    return decision_positions

def continue_until_decision_from_text(
    example_idx: int,
    text_examples: List[Dict],
    continuation_index: int,
    temperature: float,
    max_new_tokens: int,
    model=None,
    tokenizer=None,
    examples=None,
    benchmark_name=None,
    special_token_to_id=None
) -> Tuple[int, str, List[int]]:
    """
    Continue generation from a specific index in a text example until a decision token is reached.
    
    Args:
        example_idx: Index of the example
        text_examples: List of text examples
        continuation_index: Token index to start continuation from
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        model, tokenizer, examples, benchmark_name: Required for generation
        special_token_to_id: Mapping of special tokens to IDs
        
    Returns:
        Tuple of (subthought_length, generated_text, generated_token_ids)
    """
    # Find the text example
    text_example = None
    for ex in text_examples:
        if ex["example_idx"] == example_idx:
            text_example = ex
            break
    
    if text_example is None:
        raise ValueError(f"Example {example_idx} not found in text examples")
    
    # Tokenize the stored text
    stored_text = text_example["output_text"]
    stored_tokens = tokenizer.encode(stored_text, add_special_tokens=False, return_tensors="pt")[0]
    
    # Bounds-check & slice the stored tokens
    T = stored_tokens.size(0)
    k = int(max(0, min(int(continuation_index), T)))
    stored_prefix_ids = stored_tokens[:k]
    if stored_prefix_ids.numel() == 0:
        stored_prefix_ids = torch.empty((0,), dtype=torch.long, device=model.device)
    
    # Rebuild the original chat base (system+user)
    example = examples[example_idx]
    question = parse_question(example)
    messages = get_prompt(question, "qwen-instruct", benchmark_name)
    
    base_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", skip_special_tokens=False
    )[0].to(device=model.device, dtype=torch.long)
    
    # Build the full input_ids: base + stored prefix up to k
    input_ids = torch.cat([base_ids, stored_prefix_ids], dim=0)
    input_ids = input_ids.unsqueeze(0).to(dtype=torch.long)
    
    # Check input length to prevent issues
    if input_ids.size(1) > 4000:  # Reasonable limit
        print(f"Warning: Input too long ({input_ids.size(1)} tokens), skipping...")
        return max_new_tokens, "", []
    
    # Generate with specified temperature
    do_sample = temperature > 0.0
    try:
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
                attention_mask=torch.ones_like(input_ids),
                repetition_penalty=1.1,  # Prevent repetition
            )
    except Exception as e:
        print(f"Generation error: {e}")
        return max_new_tokens, "", []
    
    # Extract newly generated tokens
    seq = out.sequences[0]
    new_len = seq.size(0) - input_ids.size(1)
    gen_ids = seq[-new_len:].tolist() if new_len > 0 else []
    
    # Find first decision token in generated sequence
    special_token_ids = set(special_token_to_id.values())
    subthought_length = max_new_tokens  # Default if no decision token found
    
    for i, token_id in enumerate(gen_ids):
        if token_id in special_token_ids:
            subthought_length = i + 1  # +1 because we count the decision token
            break
    
    # Decode the generated text
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    return subthought_length, generated_text, gen_ids

def run_experiment_text(
    model_name_or_path: str,
    benchmark_name: str,
    text_dir: str,
    num_examples: int = 100,
    high_temperature: float = 1.2,
    max_new_tokens: int = 128,
    output_dir: str = "experiment_results"
) -> List[Dict]:
    """
    Run the main experiment using text-only directories.
    
    Args:
        model_name_or_path: Path to the model
        benchmark_name: Name of the benchmark dataset
        text_dir: Directory containing text output files
        num_examples: Number of examples to process
        high_temperature: Temperature for high-temperature sampling
        max_new_tokens: Maximum tokens to generate
        output_dir: Directory to save results
        
    Returns:
        List of experiment results
    """
    # Setup
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    
    # Load data
    examples = load_benchmark(benchmark_name)
    text_examples = load_text_outputs(text_dir)
    special_token_to_id = _single_token_ids_from_strings(SPECIAL_TOKENS, tokenizer)
    
    # Limit examples if specified
    if num_examples > 0:
        text_examples = text_examples[:num_examples]
    
    results = []
    
    print(f"Running experiment on {len(text_examples)} examples...")
    print(f"High temperature: {high_temperature}")
    print(f"Max new tokens: {max_new_tokens}")
    
    for text_example in tqdm(text_examples, desc="Processing examples"):
        try:
            example_idx = text_example["example_idx"]
            output_text = text_example["output_text"]
            
            # Find decision token positions in the stored text
            decision_positions = find_decision_token_positions_in_text(
                output_text, tokenizer, special_token_to_id
            )
            
            if not decision_positions:
                print(f"No decision tokens found in example {example_idx}, skipping...")
                continue
            
            # For each decision token position, run the experiment
            for decision_pos in decision_positions:
                # Skip if we're too close to the end
                if decision_pos >= len(tokenizer.encode(output_text, add_special_tokens=False)):
                    continue
                
                continuation_index = decision_pos  # Start just before the decision token
                
                # Run greedy generation
                greedy_length, greedy_text, greedy_tokens = continue_until_decision_from_text(
                    example_idx=example_idx,
                    text_examples=text_examples,
                    continuation_index=continuation_index,
                    temperature=0.0,  # Greedy
                    max_new_tokens=max_new_tokens,
                    model=model,
                    tokenizer=tokenizer,
                    examples=examples,
                    benchmark_name=benchmark_name,
                    special_token_to_id=special_token_to_id
                )
                
                # Run high-temperature generation
                high_temp_length, high_temp_text, high_temp_tokens = continue_until_decision_from_text(
                    example_idx=example_idx,
                    text_examples=text_examples,
                    continuation_index=continuation_index,
                    temperature=high_temperature,
                    max_new_tokens=max_new_tokens,
                    model=model,
                    tokenizer=tokenizer,
                    examples=examples,
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

# Import analysis functions from the main experiment
from experiment_subthought_length import analyze_results, create_visualizations

def main():
    parser = argparse.ArgumentParser(description="Subthought Length Temperature Experiment (Text Version)")
    parser.add_argument('--model_name_or_path', required=True, help="Path to the model")
    parser.add_argument('--benchmark_name', type=str, default="math", help="Benchmark dataset name")
    parser.add_argument('--text_dir', required=True, help="Directory containing text output files")
    parser.add_argument('--num_examples', type=int, default=100, help="Number of examples to process")
    parser.add_argument('--high_temperature', type=float, default=1.2, help="High temperature for sampling")
    parser.add_argument('--max_new_tokens', type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument('--output_dir', type=str, default="experiment_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Run the experiment
    results = run_experiment_text(
        model_name_or_path=args.model_name_or_path,
        benchmark_name=args.benchmark_name,
        text_dir=args.text_dir,
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
