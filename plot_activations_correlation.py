import argparse
from pathlib import Path
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


import torch
from transformers import AutoTokenizer

from utils.activations_loader import load_activations_idx, get_model_activations
from analysis.activations_analysis import compute_cosine_similarity_pairwise
from analysis.plotting_utils import plot_similarity_matrix

parser = argparse.ArgumentParser()

# Required args
parser.add_argument('--model_name_or_path')
parser.add_argument('--activations_dir')
parser.add_argument('--plot_dir')

# Optional args
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1)

parser.add_argument('--do_euclidean', type=bool, default=True)
parser.add_argument('--do_correlation', type=bool, default=True)
parser.add_argument('--do_cosine', type=bool, default=True)
parser.add_argument('--do_pca_cosine', type=bool, default=True)

args = parser.parse_args()


SPECIAL_TOKENS = {"So", "Let", "Hmm", "I", "Okay", "First", "Wait", "But", "Now", "Then",
                  "Since", "Therefore", "If", "Maybe", "To"}

def compute_cos_sim_pairwise():
    # Normalize rows and compute cosine similarity
    Xn = torch.nn.functional.normalize(X, p=2, dim=1, eps=1e-8)
    S = Xn @ Xn.t()   # (N, N)
    return S


def plot_adjacent_similarity(
    values: torch.Tensor, # similarity matrix
    tokens: list, 
    save_path: Path,
):
    X = values.to(dtype=torch.float32, device="cpu")

    N = X.shape[0]

    # --- collect series ---
    xs = []
    adj_vals = []
    best_vals = []
    best_idx = []

    adj_hover = []
    best_hover = []

    for i in range(N - 2):
        j = i + 1
        k = i + 2

        adj = float(X[i, j])
        # best non-adjacent starting at i+2 (skip i and i+1 by construction)
        slice_vals = X[i, k:]
        local = int(np.argmax(slice_vals))
        best_j = k + local
        best_v = float(slice_vals[local])

        xs.append(i)
        adj_vals.append(adj)
        best_vals.append(best_v)
        best_idx.append(best_j)

        adj_hover.append(
            f"i={i} ({tokens[i]}) ↔ j={j} ({tokens[j]})<br>"
            f"adjacent cos={adj:.4f}"
        )
        best_hover.append(
            f"i={i} ({tokens[i]}) → best={best_j} ({tokens[best_j]})<br>"
            f"best non-adj cos={best_v:.4f}"
        )

    # --- plotly figure ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xs, y=adj_vals, mode="lines+markers",
        name="Adjacent cosine (i, i+1)",
        hovertemplate="%{text}<br>y=%{y:.4f}<extra></extra>",
        text=adj_hover,
    ))

    fig.add_trace(go.Scatter(
        x=xs, y=best_vals, mode="lines+markers",
        name="Best non-adjacent (i+2:)",
        hovertemplate="%{text}<br>y=%{y:.4f}<extra></extra>",
        text=best_hover,
    ))

    fig.update_layout(
        title="Adjacent Cosine vs. Best Non-Adjacent",
        xaxis_title="Adjacent pair index (i, i+1)",
        yaxis_title="Cosine similarity",
        margin=dict(l=50, r=30, t=60, b=50),
    )

    # --- Save ---
    fig.write_html(save_path, include_plotlyjs="cdn")

# compare every combination of activations temporally (num_tokens x num_tokens) comparisons
def compare_activations(activations: torch.Tensor, tokens: list, plot_dir: Path):
    num_tokens, hidden_dim = activations.size()
    if args.do_cosine:
        pairwise_cos_similarity = compute_cosine_similarity_pairwise(activations)
        print(pairwise_cos_similarity.size())
        plot_similarity_matrix(pairwise_cos_similarity, tokens, os.path.join(plot_dir, "cos_sim.png"))
        plot_adjacent_similarity(pairwise_cos_similarity, tokens, os.path.join(plot_dir, "cos_sim_simplified.html"))

def main():
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    os.makedirs(args.plot_dir, exist_ok=True)

    for example_idx in range(args.start_idx, args.end_idx):
        # Make example-level directory
        ex_dir = os.path.join(args.plot_dir, f"example_{example_idx:05d}")
        os.makedirs(ex_dir, exist_ok=True)

        # Get the activations and tokens
        activations, output_token_ids, question = load_activations_idx(args.activations_dir, example_idx)
        tokens = tokenizer.batch_decode(output_token_ids[0], skip_special_tokens=False)

        activations = torch.transpose(activations, 0, 1) # swap the token and layer dim
        # activations = activations.to(dtype=torch.float32, device="cpu")
        num_layers, num_tokens, hidden_dim = activations.size() # each layer is a new comparison

        for layer in range(num_layers):
            # Prepare layer dir
            layer_dir = os.path.join(ex_dir, f"layer_{layer:02d}")
            os.makedirs(layer_dir, exist_ok=True)

            compare_activations(activations[layer], tokens, layer_dir)

if __name__ == "__main__":
    main()
