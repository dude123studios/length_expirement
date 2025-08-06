import numpy as np

import torch
from torch.nn.functional import cosine_similarity

def compute_cosine_similarity(activations: torch.Tensor) -> torch.Tensor:
    num_tokens, num_layers, hidden_dim = activations.size()

    transitions = torch.empty(num_tokens-1, num_layers)
    for i in range(num_tokens-1):
        sim = cosine_similarity(activations[i+1], activations[i], dim=1)
        transitions[i] = sim

    # transitions = cosine_similarity(activations, activations, dim=2)

    return transitions
