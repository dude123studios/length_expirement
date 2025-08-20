import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

import gradio as gr

import torch
from transformers import AutoTokenizer

from utils.activations_loader import load_activations_idx, get_model_activations
from analysis.mi_analysis_optimized import compute_mi_batch
from analysis.plotting_utils import get_cosine_similarity_layer_by_layer, get_token_cosine_similarity_colored_html


parser = argparse.ArgumentParser()

# Required args
parser.add_argument('--model_name_or_path')
parser.add_argument('--activations_dir')

# Optional args
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1)


args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
analysis_methods = {"Cosine Similarity":"cos", "Mutual Information (HSIC)":"mi"}

def load_example(example_idx: int):
    """
    Load the activations and tokens for a given example index (relative to args.start_idx).
    Returns:
      - tokens: list[str]
      - final_layer_reprs: Tensor [num_tokens, hidden_size]
    """
    index = args.start_idx + int(example_idx)
    activations, output_token_ids, question = load_activations_idx(args.activations_dir, index)

    tokens = tokenizer.convert_ids_to_tokens(output_token_ids[0], skip_special_tokens=True)

    return activations, tokens, question

def compute_and_render(example_number: int, ground_truth_text: str, apply_chat_template: bool):
    """
    Called by the UI. Loads the example, obtains GT rep via provided function,
    computes MI per-token against GT, returns (token_html, plot_figure, label_text).
    """
    example_activations, tokens, question = load_example(example_number)
    num_tokens, num_layers, hidden_dim = example_activations.size()
    example_activations = example_activations[:, -1, :]  # take final layer for all tokens

    # TODO: support chat template more generally for benchmarks
    if apply_chat_template == True:
        messages = get_prompt(question, 'qwen-instruct', 'math')
        mesasges.append({"role": "assistant", "content": ground_truth_text})
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
    else:
        batch_encoding = tokenizer(ground_truth_text, return_tensors="pt")
        input_ids = batch_encoding["input_ids"]

    # --- get ground-truth final-layer rep at last token ---
    gt_activations = get_model_activations(args.model_name_or_path, input_ids) # [T, H]
    gt_activations = gt_activations[-1, :] # [H] take the final token and layer representation
    gt_activations = gt_activations.unsqueeze(0).expand(num_tokens, -1) # [T, H] batch it along token dimension

    # --- MI per token vs GT ---
    mi_scores = compute_mi_batch(
        example_activations, # [T, H]
        gt_activations, # [T, H]
    )

    # --- Token heatmap HTML (reusing your existing function; treat MI like a "similarity") ---
    token_html = get_token_cosine_similarity_colored_html(
        mi_scores.cpu(),
        tokens,
        normalize=True
    )

    # --- Line plot of MI over tokens ---
    fig = plt.figure(figsize=(10, 3))
    ax = fig.gca()
    ax.plot(range(len(mi_scores)), mi_scores.cpu().numpy())
    ax.set_xlabel("Token index")
    ax.set_ylabel("Mutual Information (HSIC)")
    ax.set_title("MI between each token's final-layer rep and GT representation")
    ax.grid(True, linestyle="--", alpha=0.4)

    label = f"Example {example_number}"

    return token_html, fig, label

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 LLM Activation Viewer: Token–Ground Truth MI")

    example_input = gr.Number(value=0, precision=0, label="Example number (relative to start_idx)")
    gt_text = gr.Textbox(lines=2, label="Ground-truth solution text")
    template_checkbox = gr.Checkbox(label="Apply chat template with question to solution text", value=False)
    run_btn = gr.Button("Compute MI")

    label = gr.Text(label="Current Example", value="Example 0")
    plot_display = gr.Plot(label="MI vs. Token (final layer)", format="png")
    token_html = gr.HTML(label="Tokens (colored by MI)")


    run_btn.click(
        fn=lambda ex, txt, chat: compute_and_render(int(ex), txt, chat),
        inputs=[example_input, gt_text, template_checkbox],
        outputs=[token_html, plot_display, label]
    )

if __name__ == "__main__":
    demo.launch(share=True)
