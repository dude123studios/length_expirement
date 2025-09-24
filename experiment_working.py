#!/usr/bin/env python3
"""
Working Subthought Length Temperature Experiment
Based on the proven working patterns from plot_prob_decision_token_continuation.py
"""

import argparse
import torch
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import json
import os
from typing import List, Dict, Tuple

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
    """Load text outputs from a directory containing example_*_output.txt files."""
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

def find_decision_tokens_in_text(text: str, tokenizer, special_token_to_id: Dict[str, int]) -> List[int]:
    """Find positions of decision tokens in the text."""
    tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
    decision_positions = []
    special_token_ids = set(special_token_to_id.values())
    
    for i, token_id in enumerate(tokens.tolist()):
        if token_id in special_token_ids:
            decision_positions.append(i)
    
    return decision_positions

def generate_continuation_working(
    example_idx: int,
    text_examples: List[Dict],
    continuation_index: int,
    temperature: float,
    max_new_tokens: int,
    model,
    tokenizer,
    examples,
    benchmark_name: str,
    special_token_to_id: Dict[str, int]
) -> Tuple[int, str]:
    """
    Generate continuation using the EXACT pattern from working examples.
    """
    # Find the text example
    text_example = None
    for ex in text_examples:
        if ex["example_idx"] == example_idx:
            text_example = ex
            break
    
    if text_example is None:
        return max_new_tokens, ""
    
    # Tokenize the stored text
    stored_text = text_example["output_text"]
    stored_tokens = tokenizer.encode(stored_text, add_special_tokens=False, return_tensors="pt")[0]
    
    # Bounds-check & slice the stored tokens
    T = stored_tokens.size(0)
    k = int(max(0, min(int(continuation_index), T)))
    stored_prefix_ids = stored_tokens[:k]
    
    # Rebuild the original chat base (system+user) - EXACT pattern from working code
    example = examples[example_idx]
    question = parse_question(example)
    messages = get_prompt(question, "qwen-instruct", benchmark_name)
    
    base_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", skip_special_tokens=False
    )[0]
    
    # Ensure all tensors are on the same device before concatenation
    device = model.device
    base_ids = base_ids.to(device=device, dtype=torch.long)
    stored_prefix_ids = stored_prefix_ids.to(device=device, dtype=torch.long)
    
    # Build the full input_ids: base + stored prefix up to k
    input_ids = torch.cat([base_ids, stored_prefix_ids], dim=0)
    input_ids = input_ids.unsqueeze(0).to(dtype=torch.long)
    
    # Check input length
    if input_ids.size(1) > 2000:
        return max_new_tokens, ""
    
    # Generate using EXACT pattern from working examples
    try:
        with torch.no_grad():
            # Use ONLY the basic generation parameters that work
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=int(max_new_tokens),
                do_sample=temperature > 0.0,  # Only set do_sample based on temperature
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    except Exception as e:
        print(f"Generation error: {e}")
        return max_new_tokens, ""
    
    # Extract newly generated tokens - EXACT pattern from working code
    seq = out.sequences[0]
    new_len = seq.size(0) - input_ids.size(1)
    gen_ids = seq[-new_len:] if new_len > 0 else seq[:0]
    
    # Find first decision token in generated sequence
    special_token_ids = set(special_token_to_id.values())
    subthought_length = max_new_tokens  # Default if no decision token found
    
    for i, token_id in enumerate(gen_ids.tolist()):
        if token_id in special_token_ids:
            subthought_length = i + 1
            break
    
    # Decode the generated text
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    return subthought_length, generated_text

def run_working_experiment(
    model_name_or_path: str,
    benchmark_name: str,
    text_dir: str,
    num_examples: int = 10,
    high_temperature: float = 2.5,
    max_new_tokens: int = 128,
    output_dir: str = "working_results"
) -> List[Dict]:
    """Run the working experiment using proven patterns."""
    
    print(f"🚀 Working Subthought Length Experiment")
    print(f"Setting up model: {model_name_or_path}")
    
    # Setup - EXACT pattern from working examples
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    device = model.device
    
    print(f"Model loaded successfully on device: {device}")
    
    # Load data
    examples = load_benchmark(benchmark_name)
    text_examples = load_text_outputs(text_dir)
    special_token_to_id = _single_token_ids_from_strings(SPECIAL_TOKENS, tokenizer)
    
    # Limit examples
    if num_examples > 0:
        text_examples = text_examples[:num_examples]
    
    results = []
    
    print(f"Running experiment on {len(text_examples)} examples...")
    
    for text_example in tqdm(text_examples, desc="Processing examples"):
        try:
            example_idx = text_example["example_idx"]
            output_text = text_example["output_text"]
            
            # Find decision token positions
            decision_positions = find_decision_tokens_in_text(
                output_text, tokenizer, special_token_to_id
            )
            
            if not decision_positions:
                print(f"No decision tokens found in example {example_idx}, skipping...")
                continue
            
            # For each decision token position, run the experiment
            for i, decision_pos in enumerate(decision_positions):
                if i >= 2:  # Limit to first 2 decision tokens
                    break
                
                # Start from the decision token position
                continuation_index = decision_pos
                
                print(f"Processing example {example_idx}, decision position {decision_pos}")
                
                # Run greedy generation
                greedy_length, greedy_text = generate_continuation_working(
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
                
                print(f"Greedy generation completed: length={greedy_length}")
                
                # Run high-temperature generation
                high_temp_length, high_temp_text = generate_continuation_working(
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
                
                print(f"High-temp generation completed: length={high_temp_length}")
                
                # Store results
                result = {
                    "example_idx": example_idx,
                    "decision_position": decision_pos,
                    "greedy_length": greedy_length,
                    "high_temp_length": high_temp_length,
                    "length_difference": high_temp_length - greedy_length,
                    "greedy_text": greedy_text[:100],
                    "high_temp_text": high_temp_text[:100],
                    "temperature": high_temperature
                }
                results.append(result)
                
                print(f"✅ Example {example_idx}: Greedy={greedy_length}, High-temp={high_temp_length}")
                
        except Exception as e:
            print(f"❌ Error processing example {example_idx}: {e}")
            continue
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"working_results_{benchmark_name}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Experiment completed. Results saved to {results_file}")
    print(f"Total comparisons: {len(results)}")
    
    return results

def analyze_working_results(results: List[Dict]) -> Dict:
    """Analyze results."""
    if not results:
        return {}
    
    greedy_lengths = [r['greedy_length'] for r in results]
    high_temp_lengths = [r['high_temp_length'] for r in results]
    differences = [r['length_difference'] for r in results]
    
    stats_dict = {
        "n_comparisons": len(results),
        "greedy_mean": np.mean(greedy_lengths),
        "greedy_std": np.std(greedy_lengths),
        "high_temp_mean": np.mean(high_temp_lengths),
        "high_temp_std": np.std(high_temp_lengths),
        "difference_mean": np.mean(differences),
        "difference_std": np.std(differences)
    }
    
    # Simple t-test
    if len(results) > 1:
        t_stat, t_pvalue = stats.ttest_rel(high_temp_lengths, greedy_lengths)
        stats_dict["t_test_p_value"] = t_pvalue
    else:
        stats_dict["t_test_p_value"] = None
    
    return stats_dict

def main():
    parser = argparse.ArgumentParser(description="Working Subthought Length Temperature Experiment")
    parser.add_argument('--model_name_or_path', required=True, help="Path to the model")
    parser.add_argument('--benchmark_name', type=str, default="math", help="Benchmark dataset name")
    parser.add_argument('--text_dir', required=True, help="Directory containing text output files")
    parser.add_argument('--num_examples', type=int, default=10, help="Number of examples to process")
    parser.add_argument('--high_temperature', type=float, default=2.5, help="High temperature for sampling")
    parser.add_argument('--max_new_tokens', type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument('--output_dir', type=str, default="working_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Run the experiment
    results = run_working_experiment(
        model_name_or_path=args.model_name_or_path,
        benchmark_name=args.benchmark_name,
        text_dir=args.text_dir,
        num_examples=args.num_examples,
        high_temperature=args.high_temperature,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir
    )
    
    # Analyze results
    stats = analyze_working_results(results)
    
    print("\n📊 Analysis Results:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ Working experiment completed successfully!")

if __name__ == "__main__":
    main()
