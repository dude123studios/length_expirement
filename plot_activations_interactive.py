import argparse
import os
import json
from tqdm import tqdm

from utils.activations_loader import load_activations_idx
from analysis.activations_analysis import compute_cosine_similarity
from analysis.plotting_utils import get_cosine_similarity_layer_by_layer, get_token_cosine_similarity_colored_html

from transformers import AutoTokenizer
import gradio as gr

parser = argparse.ArgumentParser()

# Required args
parser.add_argument('--model_name_or_path')
parser.add_argument('--activations_dir')

# Optional args
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1)


args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

# === Core logic
def load_example(example_idx, layer_idx, average_layers):
    index = args.start_idx + example_idx
    activations, output_token_ids = load_activations_idx(args.activations_dir, index)
    transitions = compute_cosine_similarity(activations)  # shape: [tokens-1, layers]
    tokens = tokenizer.convert_ids_to_tokens(output_token_ids[0], skip_special_tokens=True)

    if average_layers:
        # Average over layers: [num_tokens-1, num_layers-1] -> [num_tokens-1]
        similarities = transitions[:,:-1].mean(dim=1)
    else:
        similarities = transitions[:, layer_idx]

    token_html = get_token_cosine_similarity_colored_html(similarities, tokens)
    figure = get_cosine_similarity_layer_by_layer(transitions, model_name=args.model_name_or_path)

    return token_html, figure, f"Example {example_idx}", transitions.size(1)

def update_layer(example_idx, layer_idx, average_layers):
    token_html, image_buf, label, _ = load_example(example_idx, layer_idx, average_layers)
    return token_html, image_buf, label, example_idx

def update_example(example_idx, layer_idx, average_layers, direction):
    new_idx = (example_idx + direction) % (args.end_idx - args.start_idx) 
    token_html, image_buf, label, _ = load_example(new_idx, layer_idx, average_layers)
    return token_html, image_buf, label, new_idx

# === Initial detection of layer count
_, _, _, num_layers = load_example(args.start_idx, 0, False)
layer_choices = [str(i) for i in range(num_layers)]

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 LLM Activation Viewer: Token & Layer Cosine Similarity")

    state_example_idx = gr.State(value=0)

    with gr.Row():
        label = gr.Text(label="Current Example")
        prev_button = gr.Button("← Previous Example")
        next_button = gr.Button("Next Example →")

    with gr.Row():
        layer_dropdown = gr.Dropdown(label="Layer", choices=layer_choices, value="0", interactive=True)
        mean_checkbox = gr.Checkbox(label="Average over all layers", value=False)


    plot_display = gr.Plot(label="Cosine Similarity (Layer by Layer)", format="png")
    token_html = gr.HTML(label="Tokens (colored by cosine similarity)")

    # === Interactivity
    prev_button.click(
        fn=lambda idx, layer, avg: update_example(idx, int(layer), avg, -1),
        inputs=[state_example_idx, layer_dropdown, mean_checkbox],
        outputs=[token_html, plot_display, label, state_example_idx]
    )

    next_button.click(
        fn=lambda idx, layer, avg: update_example(idx, int(layer), avg, 1),
        inputs=[state_example_idx, layer_dropdown, mean_checkbox],
        outputs=[token_html, plot_display, label, state_example_idx]
    )

    layer_dropdown.change(
        fn=lambda idx, layer, avg: update_layer(idx, int(layer), avg),
        inputs=[state_example_idx, layer_dropdown, mean_checkbox],
        outputs=[token_html, plot_display, label, state_example_idx]
    )

    mean_checkbox.change(
        fn=lambda idx, layer, avg: update_layer(idx, int(layer), avg),
        inputs=[state_example_idx, layer_dropdown, mean_checkbox],
        outputs=[token_html, plot_display, label, state_example_idx]
    )

    # Preload first example
    token_html, plot_display, label, _ = load_example(args.start_idx, 0, True)
    #token_html.update(token_html_default)
    #plot_display.update(image_buf_default)
    #label.update(label_default)


if __name__ == "__main__":
    demo.launch(share=True)
