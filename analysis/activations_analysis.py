import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import torch
from torch.nn.functional import cosine_similarity

def compute_cosine_similarity(activations: torch.Tensor) -> torch.Tensor:
    num_tokens, num_layers, hidden_dim = activations.size()

    transitions = torch.empty(num_tokens-1, num_layers)
    for i in range(num_tokens-1):
        sim = cosine_similarity(activations[i+1], activations[i], dim=1)
        transitions[i] = sim

    return transitions

def plot_cosine_similarity_layer_by_layer(deltas: torch.Tensor, save_path: Path):
    deltas = np.array(deltas)
    plt.imshow(deltas.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    plt.title(f"Activation Change per Token")
    plt.ylabel("Layer")
    plt.xlabel("Token Step")
    plt.colorbar(label="Cosine Similarity")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # High-resolution image
    plt.close()  # Close the plot to avoid display in notebooks or memory buildup

def plot_cosine_similarity_layer_by_layer(
    deltas1: torch.Tensor,
    deltas2: torch.Tensor,
    save_path: Path,
    model_name_1: str = "Model 1",
    model_name_2: str = "Model 2"
):
    deltas1 = np.array(deltas1)
    plots = [deltas1]
    titles = [f"Activation Change per Token – {model_name_1}"]

    if deltas2 is not None:
        deltas2 = np.array(deltas2)
        plots.append(deltas2)
        titles.append(f"Activation Change per Token – {model_name_2}")

    fig, axs = plt.subplots(len(plots), 1, figsize=(10, 4 * len(plots)), sharex=True)

    if len(plots) == 1:
        axs = [axs]

    for i, (ax, data, title) in enumerate(zip(axs, plots, titles)):
        im = ax.imshow(data.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
        ax.set_title(title)
        ax.set_ylabel("Layer")
        if i == len(plots) - 1:
            ax.set_xlabel("Token Step")

    fig.colorbar(im, ax=axs, orientation='vertical', label="Cosine Similarity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
