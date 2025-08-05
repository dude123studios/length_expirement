import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import torch
from torch.nn.functional import cosine_similarity

from pathlib import Path

def compute_cosine_similarity(activations: torch.Tensor) -> torch.Tensor:
    num_tokens, num_layers, hidden_dim = activations.size()

    transitions = torch.empty(num_tokens-1, num_layers)
    for i in range(num_tokens-1):
        sim = cosine_similarity(activations[i+1], activations[i], dim=1)
        transitions[i] = sim

    # transitions = cosine_similarity(activations, activations, dim=2)

    return transitions

def plot_cosine_similarity_histogram(
    sims1: list,
    sims2: list,
    save_path: Path,
    model_name_1: str = "Model 1",
    model_name_2: str = "Model 2",
    title: str = "cosine similarity"
):

    # Plot histograms
    plt.figure(figsize=(8, 6))
    plt.hist(sims1, alpha=0.6, label=model_name_1, edgecolor='black', bins='auto')
    plt.hist(sims2, alpha=0.6, label=model_name_2, edgecolor='black', bins='auto')

    # Labels and legend
    plt.xlabel('cosine similarity')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_cosine_similarity_layer_by_layer(
    deltas1: torch.Tensor,
    deltas2: torch.Tensor,
    save_path: Path,
    model_name_1: str = "Model 1",
    model_name_2: str = "Model 2",
    truncate: bool = False
):
    print(deltas1.size(), deltas2.size())
    deltas1 = np.array(deltas1)
    plots = [deltas1]
    titles = [f"Activation Change per Token – {model_name_1}"]

    if deltas2 is not None:
        deltas2 = np.array(deltas2)
        plots.append(deltas2)
        titles.append(f"Activation Change per Token – {model_name_2}")

    fig, axs = plt.subplots(len(plots), 1, figsize=(10, 4 * len(plots)), sharex=truncate)

    if len(plots) == 1:
        axs = [axs]

    for i, (ax, data, title) in enumerate(zip(axs, plots, titles)):
        im = ax.imshow(data.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
        ax.set_title(title)
        ax.set_ylabel("Layer")
        if i == len(plots) - 1:
            ax.set_xlabel("Token Step")

    fig.colorbar(im, ax=axs, orientation='vertical', label="Cosine Similarity")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_token_cosine_similarity_colored(
    transitions: torch.Tensor,
    tokens: list,
    save_path: Path,
    title: str = "Cosine Similarity (Layer-Averaged)"
):
    """
    Visualize tokens colored by cosine similarity of their layer-averaged activation with the previous token.
    
    Args:
        activations (torch.Tensor): Shape [num_tokens, num_layers, hidden_dim]
        tokens (List[str]): Corresponding tokens
        save_path (Path): Where to save the plot
        title (str): Title of the plot
    """
    # Average over layers: [num_tokens-1, num_layers] -> [num_tokens-1]
    transitions_averaged = transitions.mean(dim=1)

    # [num_tokens-1] -> [num_tokens]
    sims = torch.cat([torch.tensor([1.0]), transitions_averaged])  # First token = similarity to itself

    print(f"sims {sims.size()}, tokens: {len(tokens)}")

    # Normalize similarities to [0, 1] for colormap
    # norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("RdYlGn")

    # Start plotting
    fig, ax = plt.subplots(figsize=(min(16, len(tokens) * 0.6), 2))
    ax.axis("off")
    ax.set_title(title)

    x = 0
    for i in range(len(tokens)):
        token = tokens[i]
        sim = sims[i].item()
        print(token, sim)
        color = cmap(sim)
        ax.text(x, 0.5, token, fontsize=12, color='black',
                bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3'))
        x += 1  # space between tokens

    ax.set_xlim(-0.5, x + 0.5)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
