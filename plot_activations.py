import argparse
import os
import json
from tqdm import tqdm

from utils.activations_loader import load_activations_idx
from analysis.activations_analysis import compute_cosine_similarity
from analysis.plotting_utils import plot_cosine_similarity_layer_by_layer, plot_token_cosine_similarity_colored

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()

# Required args
parser.add_argument('--model_name_or_path')
parser.add_argument('--activations_dir')
parser.add_argument('--plot_dir')

# Optional args
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1)


args = parser.parse_args()

def main():

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    for example_idx in range(args.start_idx, args.end_idx):
        activations, output_token_ids = load_activations_idx(args.activations_dir, example_idx)
        transitions = compute_cosine_similarity(activations)

        plot_cosine_similarity_layer_by_layer(
            transitions,
            None,
            os.path.join(args.plot_dir, f"example_{example_idx}_cos_similarity.png"),
            model_name_1=args.model_name_or_path,
            model_name_2=None
        )
        print('Saved cosine similarity plot')

        # plot the tokens
        output_text = tokenizer.convert_ids_to_tokens(output_token_ids[0], skip_special_tokens=True)

        plot_token_cosine_similarity_colored(
            transitions,
            output_text,
            save_path=os.path.join(args.plot_dir, f"example_{example_idx}_token_colored.png"),
            title=f"Cosine Similarity (Layer-Averaged) – Example {example_idx}"
        )
        print('Saved token plot')

if __name__ == "__main__":
    main()
