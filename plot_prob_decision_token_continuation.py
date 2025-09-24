import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import plotly.graph_objects as go
import gradio as gr

from utils.data_loader import load_benchmark, parse_question, get_prompt
from utils.activations_loader import load_activations_idx_dict
from analysis.token_analysis import SPECIAL_TOKENS  # list of token strings

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', required=True)
parser.add_argument('--benchmark_name', type=str, default="math")
parser.add_argument('--activations_dir', required=True)
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
    """Map each token string to an id iff it encodes as exactly one token."""
    mapping = {}
    for s in tokens_as_text:
        ids = tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if len(ids) == 1:
            mapping[s] = ids.item()
    return mapping

SPECIAL_TOKEN_TO_ID = _single_token_ids_from_strings(SPECIAL_TOKENS)

def fetch_stored_answer_with_tokens(example_idx: int):
    """Show stored answer and enumerate its tokens with indices & ids."""
    data = load_activations_idx_dict(args.activations_dir, example_idx)
    output_text = data["output_text"]
    output_ids = data["output_token_ids"]
    # Handle shapes: expect [1, T] or [T]
    if isinstance(output_ids, torch.Tensor):
        if output_ids.dim() == 2:
            output_ids = output_ids[0]
        output_ids = output_ids.tolist()
    # Decode per-token with skip_special_tokens=False to keep exact strings
    token_strs = tokenizer.batch_decode(torch.tensor(output_ids).unsqueeze(-1), skip_special_tokens=False)
    lines = [f"Example {example_idx} — stored model answer:\n{output_text}", "\n--- Tokenization ---"]
    for i, (tid, ts) in enumerate(zip(output_ids, token_strs)):
        safe = ts.replace("\n", "\\n")
        lines.append(f"[{i:>4}] id={tid:<7} tok={safe}")
    return "\n".join(lines)

def continue_from_index_and_plot(example_idx: int, continuation_index: int, user_prefix: str, max_new_tokens: int):
    """
    Rebuilds the original chat prompt, appends stored assistant tokens up to `continuation_index`,
    then appends `user_prefix` (if any), and continues generation greedily.
    """
    # 1) Load stored example tokens/text
    data = load_activations_idx_dict(args.activations_dir, example_idx)
    stored_ids = data["output_token_ids"]

    # Normalize to 1D Long tensor on the right device
    if isinstance(stored_ids, torch.Tensor):
        if stored_ids.dim() == 2:
            stored_ids = stored_ids[0]
        stored_ids = stored_ids.to(device=args.model_device if hasattr(args, "model_device") else model.device,
                                   dtype=torch.long)
    else:
        stored_ids = torch.tensor(stored_ids, dtype=torch.long, device=model.device)

    # Bounds-check & slice the stored assistant tokens
    T = stored_ids.numel()
    k = int(max(0, min(int(continuation_index), T)))
    stored_prefix_ids = stored_ids[:k]                                 # [k] Long
    if stored_prefix_ids.numel() == 0:
        stored_prefix_ids = torch.empty((0,), dtype=torch.long, device=model.device)

    # 2) Rebuild the original chat base (system+user) exactly like your tracing pipeline
    example = examples[example_idx]
    question = parse_question(example)
    messages = get_prompt(question, "qwen-instruct", args.benchmark_name)

    base_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", skip_special_tokens=False
    )[0].to(device=model.device, dtype=torch.long)                      # [S] Long

    # 3) Tokenize the user's extra continuation (optional)
    if user_prefix:
        user_prefix_ids = tokenizer(user_prefix, add_special_tokens=False, return_tensors="pt").input_ids[0]
        user_prefix_ids = user_prefix_ids.to(device=model.device, dtype=torch.long)  # [U] Long
    else:
        user_prefix_ids = torch.empty((0,), dtype=torch.long, device=model.device)   # [0] Long

    # 4) Build the full input_ids: base + stored prefix up to k + user prefix
    input_ids = torch.cat([base_ids, stored_prefix_ids, user_prefix_ids], dim=0)
    input_ids = input_ids.unsqueeze(0).to(dtype=torch.long)            # [1, S+k+U] Long

    # 5) Generate greedily and capture per-step scores
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,              # greedy
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 6) Decode only the newly generated continuation
    seq = out.sequences[0]
    new_len = seq.size(0) - input_ids.size(1)
    gen_ids = seq[-new_len:] if new_len > 0 else seq[:0]
    continuation_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # 7) SPECIAL TOKEN probabilities per generation step
    probs_over_steps = {label: [] for label in SPECIAL_TOKEN_TO_ID.keys()}
    for step_logits in out.scores:  # list length == # new tokens
        probs = torch.log_softmax(step_logits[0], dim=-1)
        for label, tok_id in SPECIAL_TOKEN_TO_ID.items():
            if tok_id < probs.size(0):
                probs_over_steps[label].append(float(probs[tok_id]))

    # 8) Plotly figure
    fig = go.Figure()
    x = list(range(1, len(out.scores) + 1))
    for label, series in probs_over_steps.items():
        if series:
            fig.add_trace(go.Scatter(
                x=x, y=series, mode="lines+markers", name=label,
                hovertemplate=f"step=%{{x}}<br>p('{label}')=%{{y:.4f}}"
            ))
    fig.update_layout(
        title="Next-token probabilities for SPECIAL TOKENS (continuation from stored index)",
        xaxis_title="Generation step",
        yaxis_title="Probability",
        hovermode="x unified"
    )

    header = (
        f"Example {example_idx}\n"
        f"Continuation index: {k} / {T}\n"
        f"User prefix length (tokens): {user_prefix_ids.numel()}\n"
        f"Generated {len(out.scores)} new tokens."
    )
    return f"{header}\n\nContinuation:\n{continuation_text}", fig

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## Continue from a stored example index & track SPECIAL TOKEN probabilities")

    with gr.Row():
        example_input = gr.Number(value=0, precision=0, label="Example number")
        cont_index = gr.Number(value=0, precision=0, label="Continuation index (token offset)")
        max_tokens_input = gr.Slider(minimum=1, maximum=512, value=64, step=1, label="Max new tokens")

    prefix_box = gr.Textbox(lines=3, label="(Optional) Text to append before continuing")

    with gr.Row():
        show_stored_btn = gr.Button("Show Stored Example (with token numbers)")
        run_btn = gr.Button("Generate from Index")

    stored_text = gr.Textbox(label="Stored model answer + per-token listing", lines=14)
    out_text = gr.Textbox(label="Continuation result", lines=8)
    out_plot = gr.Plot(label="SPECIAL TOKEN probabilities")

    show_stored_btn.click(
        fn=lambda ex: fetch_stored_answer_with_tokens(int(ex)),
        inputs=[example_input],
        outputs=[stored_text]
    )

    run_btn.click(
        fn=lambda ex, idx, pref, mx: continue_from_index_and_plot(int(ex), int(idx), str(pref or ""), int(mx)),
        inputs=[example_input, cont_index, prefix_box, max_tokens_input],
        outputs=[out_text, out_plot]
    )

if __name__ == "__main__":
    demo.launch(share=True)

