import os
from pathlib import Path
import json 

import torch

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def load_activations(output_dir: Path, device=None):
    data = []

    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata.jsonl found in {output_dir}")

    if device is None:
        device = get_device()

    with open(metadata_path, "r") as f:
        for line in f:
            meta = json.loads(line)

            # Load tensors
            activations_path = os.path.join(output_dir, f"example_{meta['example_idx']}_activations.pt")
            output_ids_path = os.path.join(output_dir, f"example_{meta['example_idx']}_output_ids.pt")

            activations_tensor = torch.load(activations_path, map_location=device)
            output_token_ids = torch.load(output_ids_path, map_location=device)

            data.append({
                "example_idx": meta["example_idx"],
                "question": meta["question"],
                "output_text": meta["output_text"],
                "output_token_ids": output_token_ids,
                "activations_tensor": activations_tensor
            })

    return data

def load_activations_idx(output_dir: Path, example_idx: int, device=None):
    if device is None:
        device = get_device()

    # Load tensors
    activations_path = os.path.join(output_dir, f"example_{example_idx}_activations.pt")
    output_ids_path = os.path.join(output_dir, f"example_{example_idx}_output_ids.pt")

    activations_tensor = torch.load(activations_path, map_location=device)
    output_token_ids = torch.load(output_ids_path, map_location=device)

    return activations_tensor, output_token_ids
