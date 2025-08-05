import os
from pathlib import Path
import json 

import torch

def load_activations(output_dir: Path):
    data = []

    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata.jsonl found in {output_dir}")

    with open(metadata_path, "r") as f:
        for line in f:
            meta = json.loads(line)

            # Load tensors
            activations_path = os.path.join(output_dir, f"example_{meta['example_idx']}_activations.pt")
            output_ids_path = os.path.join(output_dir, f"example_{meta['example_idx']}_output_ids.pt")

            activations_tensor = torch.load(activations_path)
            output_token_ids = torch.load(output_ids_path)

            data.append({
                "example_idx": meta["example_idx"],
                "question": meta["question"],
                "output_text": meta["output_text"],
                "output_token_ids": output_token_ids,
                "activations_tensor": activations_tensor
            })

    return data

def load_activations_idx(output_dir: Path, example_idx: int):
    # Load tensors
    activations_path = os.path.join(output_dir, f"example_{example_idx}_activations.pt")
    output_ids_path = os.path.join(output_dir, f"example_{example_idx}_output_ids.pt")

    activations_tensor = torch.load(activations_path)
    output_token_ids = torch.load(output_ids_path)

    return activations_tensor, output_token_ids
