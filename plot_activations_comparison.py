import argparse
import os
import json
from tqdm import tqdm

from utils.activations_loader import load_activations_idx
from analysis.activations_analysis import compute_cosine_similarity, plot_cosine_similarity_layer_by_layer, plot_cosine_similarity_histogram

parser = argparse.ArgumentParser()

# Required args
parser.add_argument('--model_name_or_path1', type=str, help="First model for comparison")
parser.add_argument('--model_name_or_path2', type=str, help="Second model for comparison")

parser.add_argument('--activations_dir1', type=str, help="First activation directory for comparison")
parser.add_argument('--activations_dir2', type=str, help="Second activation directory for comparison")

parser.add_argument('--plot_dir')

# Optional args
parser.add_argument('--truncate', type=bool, default=False)
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1)


args = parser.parse_args()

def main():

    sims_across_tokens1 = []
    sims_across_tokens2 = []
    mean_sims1 = []
    mean_sims2 = []

    for example_idx in tqdm(range(args.start_idx, args.end_idx)):
        activations1, output_token_ids1 = load_activations_idx(args.activations_dir1, example_idx)
        transitions1 = compute_cosine_similarity(activations1)

        activations2, output_token_ids2 = load_activations_idx(args.activations_dir2, example_idx)
        transitions2 = compute_cosine_similarity(activations2)

        sims_across_tokens1.append(transitions1.mean(dim=(1)).sum(dim=(0)))
        mean_sims1.append(transitions1.mean(dim=(0,1)))

        sims_across_tokens2.append(transitions2.mean(dim=(1)).sum(dim=(0)))
        mean_sims2.append(transitions2.mean(dim=(0,1)))

        plot_path = os.path.join(args.plot_dir, f"example_{example_idx}_cos_similarity{"" if args.truncate else "_truncated"}.png")

        plot_cosine_similarity_layer_by_layer(
            transitions1,
            transitions2,
            plot_path,
            model_name_1=args.model_name_or_path1,
            model_name_2=args.model_name_or_path2,
            truncate=args.truncate
        )

    plot_cosine_similarity_histogram(
        sims_across_tokens1,
        sims_across_tokens2,
        os.path.join(args.plot_dir, f"cos_similarity_sum_historgram.png"),
        model_name_1=args.model_name_or_path1,
        model_name_2=args.model_name_or_path2,
        title="cosine similarity summed across tokens"
    )

    plot_cosine_similarity_histogram(
        mean_sims1,
        mean_sims2,
        os.path.join(args.plot_dir, f"cos_similarity_historgram.png"),
        model_name_1=args.model_name_or_path1,
        model_name_2=args.model_name_or_path2,
        title="cosine similarity mean across tokens"
    )

if __name__ == "__main__":
    main()
