import os
from pathlib import Path
import json 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_metadata(output_dir: str):
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata.jsonl found in {output_dir}")

    with open(metadata_path, "r") as f:
        return [json.loads(line) for line in f]

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

    # get the question from the metadata
    metadata = load_metadata(output_dir)
    for meta in metadata:
        if meta.get("example_idx") == example_idx:
            question = meta["question"]
            break
    if question is None:
        raise ValueError(f"example_idx {example_idx} not found in metadata.jsonl")

    if device is None:
        device = get_device()

    # Load tensors
    activations_path = os.path.join(output_dir, f"example_{example_idx}_activations.pt")
    output_ids_path = os.path.join(output_dir, f"example_{example_idx}_output_ids.pt")

    activations_tensor = torch.load(activations_path, map_location=device)
    output_token_ids = torch.load(output_ids_path, map_location=device)

    return activations_tensor, output_token_ids, question

# TODO: pass in the model and tokenizer directly so we don't have to keep reloading it
def get_model_activations(model_name_or_path: str, input_ids: torch.Tensor, device=None):
    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)

    with torch.no_grad():
        outputs = model(
            input_ids.to(model.device),
            output_hidden_states=True,
        )

    # returns tuple (embedding, layer1, ..., final_layer)
    # shape at each position is (batch_size, num_tokens, hidden_size)
    hidden_states = outputs.hidden_states

    return hidden_states[-1][0]
