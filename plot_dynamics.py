import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import matplotlib as mpl
import string
import plotly.express as px
import pandas as pd


import gradio as gr

import torch
from transformers import AutoTokenizer

from utils.activations_loader import load_activations_idx, get_model_activations
from analysis.mi_analysis_optimized import compute_mi_batch
from analysis.mi_analysis import estimate_mi_hsic
from analysis.plotting_utils import get_cosine_similarity_layer_by_layer, get_token_cosine_similarity_colored_html


parser = argparse.ArgumentParser()

# Required args
parser.add_argument('--model_name_or_path')
parser.add_argument('--activations_dir')
parser.add_argument('--plot_dir')

# Optional args
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1)

args = parser.parse_args()


SPECIAL_TOKENS = {"So", "Let", "Hmm", "I", "Okay", "First", "Wait", "But", "Now", "Then",
                  "Since", "Therefore", "If", "Maybe", "To"}

def _normalize_token_for_match(tok: str) -> str:
    """
    Try to make tokenizer pieces comparable to plain words:
    - strip leading whitespace
    - strip common SentencePiece/BPE markers
    - strip leading punctuation
    - keep case (list is capitalized), but also try a capitalized fallback
    """
    t = tok.lstrip()  # leading spaces
    t = t.lstrip("▁Ġ")  # common markers
    t = t.lstrip(string.punctuation)
    return t

def plot_layer_pca_grid_px(
    X_layer: np.ndarray,
    tokens: list[str],
    out_dir: str,
    layer_idx: int,
    example_idx: int,
):
    """
    X_layer: (num_tokens, hidden_dim) activations for one layer
    tokens:  list of decoded tokens, len == num_tokens
    out_dir: directory to save into (will be created)
    """
    num_tokens = X_layer.shape[0]
    assert len(tokens) == num_tokens, "tokens must align with num_tokens"

    # --- PCA to 6 components ---
    pca = PCA(n_components=6)
    Z = pca.fit_transform(X_layer)            # (T, 6)
    evr = pca.explained_variance_ratio_ * 100 # (%)

    # --- Build a tidy DataFrame for Plotly Express ---
    cols = [f"PC{i+1}" for i in range(6)]
    df = pd.DataFrame(Z, columns=cols)
    df["t"] = np.arange(num_tokens)                       # ordering
    t_norm = (df["t"] - df["t"].min()) / (df["t"].max() - df["t"].min() + 1e-9)
    df["time_norm"] = t_norm                              # for color gradient
    df["token"] = tokens                                  # hover text
    # mark special tokens (robust-ish to tokenizer quirks)
    specials = []
    for tok in tokens:
        norm = _normalize_token_for_match(tok)
        specials.append("special" if (norm in SPECIAL_TOKENS or norm.capitalize() in SPECIAL_TOKENS) else "normal")
    df["kind"] = pd.Categorical(specials, categories=["normal","special"])

    # --- Nice axis labels with EVR ---
    labels = {col: f"{col} ({evr[i]:.1f}%)" for i, col in enumerate(cols)}
    labels["time_norm"] = "Token position (earlier → later)"

    # --- Scatter-matrix: 6×6 with time gradient + special markers ---
    fig = px.scatter_matrix(
        df,
        dimensions=cols,
        color="time_norm",                      # continuous → gradient over time
        symbol="kind",                          # different markers for special tokens
        symbol_map={"normal": "circle", "special": "x"},
        hover_name="token",
        labels=labels,
        title=f"Example {example_idx} • Layer {layer_idx} • PCA(6)"
    )
    fig.update_traces(diagonal_visible=False, selector=dict(type="splom"))
    # smaller markers, thin outline for contrast
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5)))

    # --- Save ---
    html_path = os.path.join(out_dir, "pca_6x6.html")
    fig.write_html(html_path, include_plotlyjs="cdn")

    # Optional PNG (requires kaleido: `pip install -U kaleido`)
    try:
        png_path = os.path.join(out_dir, "pca_6x6.png")
        fig.write_image(png_path, scale=2)
    except Exception:
        pass  # HTML is always saved; PNG is best-effort


def plot_layer_pca_grid(
    X_layer: np.ndarray,
    tokens: list,
    out_path: str,
    layer_idx: int,
    example_idx: int,
):
    """
    X_layer: (num_tokens, hidden_dim) activations for a single layer
    tokens: list[str] with length == num_tokens
    out_path: directory where the figure will be written (must exist)
    """
    num_tokens = X_layer.shape[0]
    assert len(tokens) == num_tokens, "tokens must align with num_tokens"

    # Fit PCA -> 6 comps, then transform to (num_tokens, 6)
    pca = PCA(6)
    Z = pca.fit_transform(X_layer)  # shape: (T, 6)
    evr = pca.explained_variance_ratio_  # length 6

    # Build colors over time to show dynamics
    t = np.arange(num_tokens, dtype=float)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)
    cmap = mpl.cm.get_cmap("viridis")
    colors = cmap(t_norm)

    # Split indices by "special" vs normal tokens
    special_idx = []
    normal_idx = []
    for i, tok in enumerate(tokens):
        norm = _normalize_token_for_match(tok)
        if norm in SPECIAL_TOKENS:
            special_idx.append(i)
        else:
            normal_idx.append(i)

    # Figure/axes: 6x6 grid
    fig, axes = plt.subplots(6, 6, figsize=(18, 18), constrained_layout=True)

    # Style for the trajectory path (same for all subplots)
    path_kwargs = dict(color="lightgray", linewidth=1.0, alpha=0.7)

    # Plot each pair (i, j) of PCs
    for i in range(6):
        for j in range(6):
            ax = axes[i, j]

            if i == j:
                # Diagonal: component value over time (colored by time)
                # Use scatter to preserve gradient; connect with light path
                ax.plot(np.arange(num_tokens), Z[:, i], **path_kwargs)
                ax.scatter(np.arange(num_tokens), Z[:, i], c=t_norm, cmap=cmap, s=12)

                # Emphasize special tokens with different marker
                if special_idx:
                    ax.scatter(np.array(special_idx),
                               Z[np.array(special_idx), i],
                               c=t_norm[np.array(special_idx)],
                               cmap=cmap,
                               s=36,
                               marker="X",
                               edgecolors="black",
                               linewidths=0.5)
                ax.set_xlabel(f"t")
                ax.set_ylabel(f"PC{i+1}")
            else:
                # Off-diagonal: PC_j (x) vs PC_i (y)
                # Draw a light path in sequence for visual continuity
                ax.plot(Z[:, j], Z[:, i], **path_kwargs)

                # Normal tokens
                if normal_idx:
                    idx = np.array(normal_idx)
                    ax.scatter(Z[idx, j], Z[idx, i], c=t_norm[idx], cmap=cmap, s=12)

                # Special tokens with a different marker
                if special_idx:
                    idx = np.array(special_idx)
                    ax.scatter(Z[idx, j], Z[idx, i],
                               c=t_norm[idx], cmap=cmap, s=36,
                               marker="X", edgecolors="black", linewidths=0.5)

                ax.set_xlabel(f"PC{j+1}")
                ax.set_ylabel(f"PC{i+1}")

            # Make small, neat ticks
            ax.tick_params(labelsize=8)

    # One shared colorbar for "time"
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.01)
    cbar.set_label("Token position (earlier → later)", rotation=270, labelpad=14)

    # Figure title with layer + EVR info
    evr_str = ", ".join([f"{v*100:.1f}%" for v in evr])
    fig.suptitle(
        f"Example {example_idx} • Layer {layer_idx} • PCA (6) "
        f"Explained Var: [{evr_str}]",
        fontsize=14
    )

    # Legend stub for special markers (add a tiny invisible handle)
    # (We don't add a full legend per subplot to avoid clutter.)
    # Instead, add a single annotation:
    fig.text(
        0.01, 0.01,
        "Marker:  ×  for special tokens "
        + str(sorted(list(SPECIAL_TOKENS))),
        fontsize=9
    )

    # Save
    out_file = os.path.join(out_path, "pca_6x6.png")
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def main():

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    os.makedirs(args.plot_dir, exist_ok=True)

    # TODO: concatenate hidden_states across examples and calculate global PCA components
    for example_idx in range(args.start_idx, args.end_idx):
        activations, output_token_ids, question = load_activations_idx(args.activations_dir, example_idx)
        tokens = tokenizer.batch_decode(output_token_ids[0], skip_special_tokens=False)

        activations = activations.to(dtype=torch.float32, device="cpu")
        num_tokens, num_layers, hidden_dim = activations.size()

        # Make example-level directory
        ex_dir = os.path.join(args.plot_dir, f"example_{example_idx:05d}")
        os.makedirs(ex_dir, exist_ok=True)

        for layer in range(num_layers):
            # Report variance as before (optional)
            # pca = PCA(6)
            # pca.fit(activations[:, layer, :])
            # print('explained variance: ', np.sum(pca.explained_variance_ratio_))

            # Prepare layer dir
            layer_dir = os.path.join(ex_dir, f"layer_{layer:02d}")
            os.makedirs(layer_dir, exist_ok=True)

            # Plot grid and save
            X_layer = activations[:, layer, :].numpy()
            plot_layer_pca_grid_px(
                X_layer=X_layer,
                tokens=tokens,
                out_dir=layer_dir,
                layer_idx=layer,
                example_idx=example_idx
            )

if __name__ == "__main__":
    main()
