import argparse
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import plotly.graph_objects as go
import gradio as gr

from utils.activations_loader import load_activations_idx_dict
from utils.data_loader import load_benchmark, parse_question, parse_ground_truth
from analysis.token_analysis import normalize_token_for_match, SPECIAL_TOKENS
from analysis.plotting_utils import get_token_cosine_similarity_colored_html

parser = argparse.ArgumentParser()

# Required args
parser.add_argument('--model_name_or_path', required=True)
parser.add_argument('--activations_dir', required=True)
# parser.add_argument('--plot_dir', required=True)
# Optional args
parser.add_argument('--benchmark_name', type=str, default="math")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()

examples = load_benchmark(args.benchmark_name)

def compute_and_render(example_idx: int):

    activations_data = load_activations_idx_dict(args.activations_dir, example_idx)
    model_answer = activations_data["output_text"]
    model_answer_tokens = activations_data["output_token_ids"][0]
    model_answer_tokens_str = tokenizer.batch_decode(model_answer_tokens.unsqueeze(-1), skip_special_tokens=False)

    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    example = examples[example_idx]
    question = parse_question(example)
    ground_truth_text = parse_ground_truth(example)
    # TODO: add a </think> tag
    forced_answer_text = "\nTherefore, the answer is \\boxed{"

    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    model_messages = base_messages + [{"role": "assistant", "content": model_answer}]

    tokenized_base = tokenizer.apply_chat_template(base_messages, add_generation_prompt=True, return_tensors="pt", skip_special_tokens=False).to(model.device)[0]
    # tokenized_model_response = tokenizer.apply_chat_template(model_messages, return_tensors="pt", skip_special_tokens=False).to(model.device)
    tokenized_forced_response = tokenizer(forced_answer_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)[0]
    tokenized_answer = tokenizer(ground_truth_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)[0]

    print("base length", len(tokenized_base))
    print("response length", len(model_answer_tokens))
    # print("full length", len(tokenized_model_response))

    print("tokenized_base", tokenized_base.size())
    # print("tokenized_model_response", tokenized_model_response.size())
    print("tokenized_forced_response", tokenized_forced_response.size())
    print("tokenized_answer", tokenized_answer.size())

    print("model_answer_tokens", model_answer_tokens.size())
    print("model_answer_tokens_str", model_answer_tokens_str)
    # print("model_messages", model_messages)

    # print(tokenizer.batch_decode(tokenized_model_response, skip_special_tokens=False))
    print("toknized answer", ground_truth_text, tokenizer.batch_decode(tokenized_answer.unsqueeze(-1), skip_special_tokens=False))

    answer_probs = []
    answer_log_probs = []
    entropies = []
    norm_shares = []

    # current_tokens = tokenized_base
    for t in tqdm(range(1,len(model_answer_tokens)+1)):
        # response_text = tokenizer.batch_decode(model_answer_tokens[:t], skip_special_tokens=True)[0]
        input_ids = torch.cat((tokenized_base, model_answer_tokens[:t], tokenized_forced_response, tokenized_answer))

        with torch.no_grad():
            logits = model(input_ids.unsqueeze(0)).logits[0] # shape: (seq_len, vocab_size)

        # Get the logits that predict the last A tokens
        answer_logits = logits[-len(tokenized_answer)-1:-1, :]
        # Compute softmax probabilities from logits
        log_probs = torch.log_softmax(answer_logits , dim=-1)  # [A, V]
        gathered = log_probs.gather(-1, tokenized_answer.unsqueeze(-1))  # [A, 1]
        prob_answer = gathered.squeeze(-1).sum()
        answer_log_probs.append(prob_answer.item())
        answer_probs.append(prob_answer.exp().item())

    #     # get entropy 
    #     probs = log_probs.exp()        # [A, V]
    #     entropy = -(probs * log_probs).sum(dim=-1).mean().item()
    #     entropies.append(entropy)
    #
    #     # log probs normalized by entropy
    #     norm_share = prob_answer + entropy
    #     norm_shares.append(norm_share)
    #
        # if t==100:
        #     print("tokenized splicing", tokenizer.batch_decode(input_ids, skip_special_tokens=False))
        #     print("tokenized splicing answer", tokenizer.batch_decode(input_ids[-len(tokenized_answer)-1:-1], skip_special_tokens=False))
        #
    #    print("logits size [S, V]", logits.size())
    #     print("answer logits size [A,V]", answer_logits.size())
    #     print("log probs size [A,V]", log_probs.size())
    #     print("gathered size [A]", gathered.size())
    #     print("prob_answer [1]", prob_answer)


    # Plot answer probs graph
    # --- plotly figure ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(model_answer_tokens))), y=answer_log_probs,
        mode="lines+markers", name="log-prob(answer)",
        hovertemplate="%{text}<br>logp=%{y:.4f}",
        text=model_answer_tokens_str,
    ))

    # fig.add_trace(go.Scatter(
    #     x=list(range(len(model_answer_tokens))), y=entropies,
    #     mode="lines+markers", name="entropy",
    #     hovertemplate="entropy=%{y:.4f}"
    # ))
    #
    # fig.add_trace(go.Scatter(
    #     x=list(range(len(model_answer_tokens))), y=norm_shares,
    #     mode="lines+markers", name="normalized share",
    #     hovertemplate="norm_share=%{y:.4f}"
    # ))
    #
    # special token markers
    for i, tok in enumerate(model_answer_tokens_str):
        if normalize_token_for_match(tok) in SPECIAL_TOKENS:
            fig.add_vline(x=i, line=dict(color="red", dash="dot", width=1))
            fig.add_annotation(
                x=i, y=max(answer_log_probs)*1.05,  # place above curve
                text=tok,
                showarrow=False,
                font=dict(color="red", size=10),
                yanchor="bottom"
            )

    fig.update_layout(
        title="Probability of correct solution over time",
        xaxis_title="tokens",
        yaxis_title="probabiltiy correct",
        hovermode="x unified"
    )

    # --- Token heatmap HTML ---
    token_html = get_token_cosine_similarity_colored_html(
        answer_probs,
        model_answer_tokens_str,
        normalize=False,
        pad_first_token=False,
        vmin=0.0,
        vmax=1.0
    )

    # Plot answer probs token heatmap
    return token_html, fig, example_idx


# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## Evolution of solution confidence")

    example_input = gr.Number(value=0, precision=0, label="Example number (relative to start_idx)")
    run_btn = gr.Button("Compute")

    label = gr.Text(label="Current Example", value="Example 0")
    plot_display = gr.Plot(label="Prob vs. Token")
    token_html = gr.HTML(label="Tokens (colored by probability correct)")

    run_btn.click(
        fn=lambda ex: compute_and_render(int(ex)),
        inputs=[example_input],
        outputs=[token_html, plot_display, label]
    )

if __name__ == "__main__":
    demo.launch(share=True)
