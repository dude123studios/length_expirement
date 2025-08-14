from analysis.mi_analysis_optimized import compute_mi_test

import numpy as np
from tqdm import tqdm

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

def compute_mi(activations: torch.Tensor) -> torch.Tensor:
    num_tokens, num_layers, hidden_dim = activations.size()
    print(activations.device)

    transitions = torch.zeros(num_tokens-1, num_layers)
    # for i in tqdm(range(num_tokens-1)):
        #for j in range(num_layers):
        #    sim = estimate_mi_hsic(activations[i+1,j], activations[i,j])
        #    transitions[i,j] = sim

        #sim = estimate_mi_hsic(activations[i+1,-1], activations[i,-1])
        # transitions[i,-1] = sim

    #transitions_last_layer = compute_mi_test(activations)
    # transitions[:, -1] = compute_mi_test(activations, layer="last", method="rbf") 
    # transitions = compute_mi_optimized_v2(activations)
 

    # Only calculating the last layer with non-linear rbf kernel for now 
    last_layer_scores = compute_mi_test(
        activations,            # [T, L, H]
        method="rbf",
        rbf_mode="stream",      # critical
        sigma=50.0,
        chunk_size=2048,        # tune if you see OOM
        subsample=None,         # int like 2048 to speed up
        use_double=False,       # set True for max stability
        show_progress=True,
    )
    transitions[:, -1] = last_layer_scores

    return transitions
