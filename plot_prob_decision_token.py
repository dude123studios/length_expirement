import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import plotly.graph_objects as go
import gradio as gr

from utils.data_loader import load_benchmark, parse_question
from analysis.token_analysis import SPECIAL_TOKENS  # list of token strings
from utils.activations_loader import load_activations_idx_dict

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', required=True)
parser.add_argument('--activations_dir', required=True)
parser.add_argument('--benchmark_name', type=str, default="math")
args = parser.parse_args()

# --- Model & tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16
)
model.eval()
device = model.device

# --- Data ---
examples = load_benchmark(args.benchmark_name)

def _single_token_ids_from_strings(tokens_as_text):
    """Map each token string to a single id iff it encodes as exactly one token."""
    mapping = {}
    for s in tokens_as_text:
        ids = tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if len(ids) == 1:
            mapping[s] = ids.item()
    return mapping

SPECIAL_TOKEN_TO_ID = _single_token_ids_from_strings(SPECIAL_TOKENS)


def get_model_response(example_idx: int):
    activations_data = load_activations_idx_dict(args.activations_dir, example_idx)
    model_answer = activations_data["output_text"]
    return model_answer

def continue_and_plot(example_idx: int, prefix_text: str, max_new_tokens: int):
    # --- Build chat prompt ---
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    example = examples[example_idx]
    question = parse_question(example)

    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    base_ids = tokenizer.apply_chat_template(
        base_messages, add_generation_prompt=True, return_tensors="pt", skip_special_tokens=False
    )[0].to(device)

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    input_ids = torch.cat([base_ids, prefix_ids], dim=0).unsqueeze(0)  # [1, S]

    # --- Greedy generation via .generate with score capture ---
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,                 # greedy
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,              # <-- gives per-step logits
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # New tokens (exclude the prompt/prefix)
    seq = out.sequences[0]
    new_len = seq.size(0) - input_ids.size(1)
    gen_ids = seq[-new_len:] if new_len > 0 else seq[:0]
    continuation_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # out.scores is a list (length = # new tokens) of logits at each step
    probs_over_steps = {label: [] for label in SPECIAL_TOKEN_TO_ID.keys()}
    for step_logits in out.scores:  # each: [1, vocab_size]
        probs = torch.softmax(step_logits[0], dim=-1)
        for label, tok_id in SPECIAL_TOKEN_TO_ID.items():
            # Guard for tokenizers whose vocab may not include some ids at runtime
            if tok_id < probs.size(0):
                probs_over_steps[label].append(float(probs[tok_id].item()))

    # --- Plotly chart: one line per SPECIAL TOKEN ---
    fig = go.Figure()
    x = list(range(1, len(out.scores) + 1))
    for label, series in probs_over_steps.items():
        if series:  # only plot if we recorded something
            fig.add_trace(go.Scatter(
                x=x, y=series,
                mode="lines+markers",
                name=label,
                hovertemplate=f"step=%{{x}}<br>p('{label}')=%{{y:.4f}}"
            ))

    fig.update_layout(
        title="Next-token probabilities for SPECIAL TOKENS (greedy generation)",
        xaxis_title="Generation step",
        yaxis_title="Probability",
        hovermode="x unified"
    )

    return continuation_text, fig

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## Continue from prefix & track SPECIAL TOKEN probabilities")

    example_input = gr.Number(value=0, precision=0, label="Example number")
    prefix_box = gr.Textbox(lines=3, label="Beginning of your answer (assistant prefix)")
    max_tokens_input = gr.Slider(minimum=1, maximum=512, value=64, step=1, label="Max new tokens")


    with gr.Row():
        run_btn = gr.Button("Generate")
        get_btn = gr.Button("Get model response")

    model_response_text = gr.Textbox(label="Model continuation", lines=6)
    out_text = gr.Textbox(label="Model continuation", lines=6)
    out_plot = gr.Plot(label="SPECIAL TOKEN probabilities")

    run_btn.click(
        fn=lambda ex, pref, mx: continue_and_plot(int(ex), str(pref or ""), int(mx)),
        inputs=[example_input, prefix_box, max_tokens_input],
        outputs=[out_text, out_plot]
    )

    get_btn.click(
        fn=lambda ex: get_model_response(int(ex)),
        inputs=[example_input],
        outputs=[model_response_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)

