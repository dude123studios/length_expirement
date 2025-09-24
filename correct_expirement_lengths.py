import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from analysis.token_analysis import SPECIAL_TOKENS  # list of token strings
import plotly.graph_objects as go

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', default="deepseek-ai/DeepSeek-R1-Distill-Qwen")
parser.add_argument('--activations_dir', default="/Users/atharvnaphade/Downloads/atharv/deepseek-qwen")
parser.add_argument('--max_new_tokens', type=int, default=64)  # keep small for Mac
parser.add_argument('--num_examples', type=int, default=10)
parser.add_argument('--high_temp', type=float, default=1.2)
args = parser.parse_args()

# --- Device selection ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float16
    print("Using Apple Metal (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
    print("Using CUDA")
else:
    device = torch.device("cpu")
    dtype = torch.float32
    print("Using CPU")

# --- Model & tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path, torch_dtype=dtype, device_map=None
).to(device)
model.eval()


def _single_token_ids_from_strings(tokens_as_text):
    mapping = {}
    for s in tokens_as_text:
        ids = tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if len(ids) == 1:
            mapping[s] = ids.item()
    return mapping


SPECIAL_TOKEN_TO_ID = _single_token_ids_from_strings(SPECIAL_TOKENS)


def load_trace_tokens(file_path):
    """Load a trace .txt file and tokenize it into ids."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    return text, token_ids.to(device)


def continue_until_decision(prefix_ids: torch.Tensor, temperature: float, max_new_tokens: int):
    """Continue until a SPECIAL TOKEN is hit, return subthought length."""
    input_ids = prefix_ids.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            top_p=1.0,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    seq = out.sequences[0]
    gen_ids = seq[input_ids.size(1):]

    subthought_len = len(gen_ids)
    for i, tid in enumerate(gen_ids.tolist()):
        if tid in SPECIAL_TOKEN_TO_ID.values():
            subthought_len = i + 1
            break

    continuation_text = tokenizer.decode(gen_ids[:subthought_len], skip_special_tokens=True)
    return subthought_len, continuation_text


def run_experiment():
    results = []
    trace_files = sorted(
        [os.path.join(args.activations_dir, f) for f in os.listdir(args.activations_dir) if f.endswith(".txt")]
    )

    for ex_idx, file_path in enumerate(trace_files[:args.num_examples]):
        text, token_ids = load_trace_tokens(file_path)

        decision_positions = [i for i, tid in enumerate(token_ids.tolist()) if tid in SPECIAL_TOKEN_TO_ID.values()]
        if not decision_positions:
            print(f"Skipping {file_path}: no decision token found")
            continue

        start_idx = max(0, decision_positions[0] - 5)
        prefix_ids = token_ids[:start_idx]

        greedy_len, _ = continue_until_decision(prefix_ids, temperature=0.0, max_new_tokens=args.max_new_tokens)
        sampled_len, _ = continue_until_decision(prefix_ids, temperature=args.high_temp, max_new_tokens=args.max_new_tokens)

        results.append({
            "file": os.path.basename(file_path),
            "greedy_len": greedy_len,
            "sampled_len": sampled_len,
        })
        print(f"{os.path.basename(file_path)} → greedy={greedy_len}, sampled={sampled_len}")

    if not results:
        print("⚠️ No results collected — check your SPECIAL_TOKENS and traces folder.")
        return

    greedy_avg = np.mean([r["greedy_len"] for r in results])
    sampled_avg = np.mean([r["sampled_len"] for r in results])

    print("\n=== Results ===")
    print(f"Greedy avg length: {greedy_avg:.2f}")
    print(f"Sampled avg length (T={args.high_temp}): {sampled_avg:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Box(y=[r["greedy_len"] for r in results], name="Greedy"))
    fig.add_trace(go.Box(y=[r["sampled_len"] for r in results], name=f"Sampled (T={args.high_temp})"))
    fig.update_layout(title="Subthought Lengths until Decision Token", yaxis_title="Tokens")
    fig.show()


if __name__ == "__main__":
    run_experiment()
