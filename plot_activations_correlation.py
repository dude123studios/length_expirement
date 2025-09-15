import argparse
from pathlib import Path
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from torch.nn.functional import cosine_similarity

from utils.activations_loader import load_activations_idx, get_model_activations
from analysis.activations_analysis import compute_cosine_similarity_pairwise
from analysis.plotting_utils import plot_similarity_matrix
from analysis.token_analysis import normalize_token_for_match, SPECIAL_TOKENS

parser = argparse.ArgumentParser()

# Required args
parser.add_argument('--model_name_or_path')
parser.add_argument('--activations_dir')
parser.add_argument('--plot_dir')

# Optional args
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1)

parser.add_argument('--do_cosine', type=bool, default=False)
parser.add_argument('--do_pca_cosine', type=bool, default=False)
parser.add_argument('--do_momentum', type=bool, default=False)
parser.add_argument('--do_pca_momentum', type=bool, default=False)
parser.add_argument('--do_pca_momentum_components', type=bool, default=False)

args = parser.parse_args()


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
    markers = []

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
            f"{tokens[i]} ↔ {tokens[j]}<br>"
        )
        best_hover.append(
            f"{tokens[i]} → {tokens[best_j]}<br>"
            f"{i} ↔ {best_j}<br>"
        )

        markers.append("square" if tokens[i] in SPECIAL_TOKENS else "circle")

    # --- plotly figure ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xs, y=adj_vals, mode="lines+markers",
        name="Adjacent cosine (i, i+1)",
        hovertemplate="%{text}<br>y=%{y:.4f}<extra></extra>",
        text=adj_hover,
        marker=dict(symbol=markers, size=9)
    ))

    fig.add_trace(go.Scatter(
        x=xs, y=best_vals, mode="lines+markers",
        name="Best non-adjacent (i+2:)",
        hovertemplate="%{text}<br>y=%{y:.4f}<extra></extra>",
        text=best_hover,
        marker=dict(symbol=markers, size=9)
    ))

    fig.update_layout(
        title="Adjacent Cosine vs. Best Non-Adjacent",
        xaxis_title="Adjacent pair index (i, i+1)",
        yaxis_title="Cosine similarity",
        margin=dict(l=50, r=30, t=60, b=50),
    )

    # --- Save ---
    fig.write_html(save_path, include_plotlyjs="cdn")

def plot_momentum(
    activations: torch.Tensor, # (num_tokens, num_features) / (N,d)
    tokens: list, 
    save_path: Path,
    smooth_k: int = 10,
):
    num_tokens, num_features = activations.size()
    velocities = activations[1:,:] - activations[:-1, :] # N-1 velocities / (N-1, d)

    speeds = torch.norm(velocities, dim=1) # (N-1)

    direction_coalignment = cosine_similarity(velocities[:-1], velocities[1:], dim=1)

    speeds = speeds.to(dtype=torch.float32, device="cpu")
    direction_coalignment = direction_coalignment .to(dtype=torch.float32, device="cpu")

    # smooth out the speed
    if smooth_k > 1:
        speeds = np.convolve(speeds, np.ones(smooth_k), 'valid') / smooth_k
        direction_coalignment = np.convolve(direction_coalignment, np.ones(smooth_k), 'valid') / smooth_k

    # normalize smoothed speeds
    speeds = (speeds - speeds.min()) / (speeds.max() - speeds.min() + 1e-9) # normalize speeds from 0 to 1 

    # --- plotly figure ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(speeds))), y=speeds, mode="lines+markers",
        name="speed",
        hovertemplate="%{text}<br>y=%{y:.4f}<extra></extra>",
        text=tokens[:-1],
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(direction_coalignment))), y=direction_coalignment, mode="lines+markers",
        name="direction cos-sim",
    ))

    # special token markers
    for i, tok in enumerate(tokens):
        if normalize_token_for_match(tok) in SPECIAL_TOKENS and i < num_tokens-1:
            fig.add_vline(x=i, line=dict(color="red", dash="dot", width=1))
            fig.add_annotation(
                x=i, y=max(speeds)*1.05,  # place above curve
                text=tok,
                showarrow=False,
                font=dict(color="red", size=10),
                yanchor="bottom"
            )

    fig.update_layout(
        title="Activation speed and direction alignment",
        xaxis_title="tokens",
        yaxis_title="velocity",
        hovermode="x unified"
    )

    # --- Save ---
    fig.write_html(save_path, include_plotlyjs="cdn")

# compare every combination of activations temporally (num_tokens x num_tokens) comparisons
def compare_activations(activations: torch.Tensor, tokens: list, plot_dir: Path):
    num_tokens, hidden_dim = activations.size()
    if args.do_cosine:
        pairwise_cos_similarity = compute_cosine_similarity_pairwise(activations)
        plot_similarity_matrix(pairwise_cos_similarity, tokens, os.path.join(plot_dir, "cos_sim.png"))
        plot_adjacent_similarity(pairwise_cos_similarity, tokens, os.path.join(plot_dir, "cos_sim_simplified.html"))
    if args.do_pca_cosine:
        pca = PCA(n_components=0.95)
        activations_feature_reducded = pca.fit_transform(activations.to(dtype=torch.float32, device="cpu"))
        activations_feature_reducded = torch.tensor(activations_feature_reducded)
        reduced_pairwise_cos_similarity = compute_cosine_similarity_pairwise(activations_feature_reducded)
        plot_similarity_matrix(reduced_pairwise_cos_similarity, tokens, os.path.join(plot_dir, "cos_sim_pca.png"))
        plot_adjacent_similarity(reduced_pairwise_cos_similarity, tokens, os.path.join(plot_dir, "cos_sim_simplified_pca.html"))
    if args.do_momentum:
        plot_momentum(activations, tokens, os.path.join(plot_dir, "momentum.html"))
    if args.do_pca_momentum:
        pca = PCA(n_components=0.95)
        activations_feature_reducded = pca.fit_transform(activations.to(dtype=torch.float32, device="cpu"))
        activations_feature_reducded = torch.tensor(activations_feature_reducded)
        plot_momentum(activations_feature_reducded, tokens, os.path.join(plot_dir, "momentum_pca.html"))
    if args.do_pca_momentum_components:
        pca = PCA(n_components=6)
        activations_feature_reducded = pca.fit_transform(activations.to(dtype=torch.float32, device="cpu"))
        activations_feature_reducded = torch.tensor(activations_feature_reducded)
        for i in range(6):
            plot_momentum(activations_feature_reducded[:,i].unsqueeze(1), tokens, os.path.join(plot_dir, f"momentum_pca_{i}.html"))


def main():
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    os.makedirs(args.plot_dir, exist_ok=True)

    for example_idx in tqdm(range(args.start_idx, args.end_idx)):
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
