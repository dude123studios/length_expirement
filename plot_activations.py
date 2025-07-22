import argparse
import os
import json
from tqdm import tqdm

from utils.activations_loader import load_activations_idx
from analysis.activations_analysis import compute_cosine_similarity, plot_cosine_similarity_layer_by_layer

parser = argparse.ArgumentParser()

# Required args
parser.add_argument('--activations_dir')
parser.add_argument('--plot_dir')
# Optional args
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=-1)


args = parser.parse_args()

def main():

    for example_idx in range(args.start_idx, args.end_idx):
        activations, output_token_ids = load_activations_idx(args.activations_dir, example_idx)
        transitions = compute_cosine_similarity(activations)
        plot_cosine_similarity_layer_by_layer(transitions, os.path.join(args.plot_dir, f"example_{example_idx}_cos_similarity.png"))

if __name__ == "__main__":
    main()
