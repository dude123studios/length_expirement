#!/usr/bin/env python3
"""
Convert text output files to proper activations format for experiments.

This script takes directories containing only text output files and converts them
to the full activations format expected by the experiment scripts.
"""

import argparse
import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

def convert_text_directory_to_activations(
    text_dir: str,
    output_dir: str,
    model_name_or_path: str,
    benchmark_name: str = "math"
):
    """
    Convert a directory of text files to proper activations format.
    
    Args:
        text_dir: Directory containing example_*_output.txt files
        output_dir: Directory to save the converted activations format
        model_name_or_path: Model path for tokenization
        benchmark_name: Benchmark name for metadata
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # Find all text files
    text_files = sorted([f for f in os.listdir(text_dir) if f.startswith("example_") and f.endswith("_output.txt")])
    
    if not text_files:
        raise ValueError(f"No example_*_output.txt files found in {text_dir}")
    
    print(f"Found {len(text_files)} text files to convert...")
    
    metadata_entries = []
    
    for text_file in tqdm(text_files, desc="Converting files"):
        # Extract example index
        example_idx = int(text_file.replace("example_", "").replace("_output.txt", ""))
        
        # Read text content
        text_path = os.path.join(text_dir, text_file)
        with open(text_path, 'r', encoding='utf-8') as f:
            output_text = f.read().strip()
        
        # Tokenize the text
        output_token_ids = tokenizer.encode(output_text, add_special_tokens=False, return_tensors="pt")
        
        # Create dummy activations tensor (we'll need to regenerate these properly)
        # For now, create a placeholder tensor with the right shape
        num_tokens = output_token_ids.size(1)
        num_layers = 32  # Typical for 7B models
        hidden_dim = 4096  # Typical for 7B models
        
        # Create dummy activations (zeros for now - these would need to be regenerated)
        activations_tensor = torch.zeros((num_tokens, num_layers-1, hidden_dim), dtype=torch.float16)
        
        # Save activations tensor
        activations_path = os.path.join(output_dir, f"example_{example_idx}_activations.pt")
        torch.save(activations_tensor, activations_path)
        
        # Save output token IDs
        output_ids_path = os.path.join(output_dir, f"example_{example_idx}_output_ids.pt")
        torch.save(output_token_ids, output_ids_path)
        
        # Save output text (copy)
        output_text_path = os.path.join(output_dir, f"example_{example_idx}_output.txt")
        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        # Create metadata entry
        metadata_entry = {
            "example_idx": example_idx,
            "question": f"Question {example_idx}",  # Placeholder - would need actual questions
            "output_text": output_text,
            "output_token_ids": output_token_ids.squeeze().tolist()
        }
        metadata_entries.append(metadata_entry)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Conversion complete!")
    print(f"Converted {len(text_files)} examples")
    print(f"Output saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Convert text outputs to activations format")
    parser.add_argument('--text_dir', required=True, help="Directory containing text output files")
    parser.add_argument('--output_dir', required=True, help="Directory to save converted activations")
    parser.add_argument('--model_name_or_path', required=True, help="Model path for tokenization")
    parser.add_argument('--benchmark_name', default="math", help="Benchmark name")
    
    args = parser.parse_args()
    
    convert_text_directory_to_activations(
        text_dir=args.text_dir,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        benchmark_name=args.benchmark_name
    )

if __name__ == "__main__":
    main()

