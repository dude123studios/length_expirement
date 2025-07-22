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

