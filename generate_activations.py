import argparse
import os
import json
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

from utils.arg_parser import SamplingParams
from utils.data_loader import load_benchmark, parse_question, get_prompt

parser = argparse.ArgumentParser()

# Required args
parser.add_argument('--model_name_or_path')
parser.add_argument('--benchmark_name')
parser.add_argument('--sampling_config', type=str)

parser.add_argument('--output_dir')
# Optional args
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=-1)
# Sampling overrides (more otipnal args)
parser.add_argument('--prompt_style')
parser.add_argument('--temperature', type=float)
parser.add_argument('--top_p', type=float)
parser.add_argument('--k', type=int)
parser.add_argument('--n_sampling', type=int)
parser.add_argument('--max_tokens', type=int)
parser.add_argument('--seed', type=int)

args = parser.parse_args()

# Load args and merge yaml args
sampling_params = SamplingParams.from_args_and_yaml(args, args.sampling_config)

print("Final Sampling Parameters:", sampling_params)

def main():

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)
    # generation_config = GenerationConfig.from_pretrained(
    #     args.model_name_or_path,
    # )

    # load dataset
    examples = load_benchmark(args.benchmark_name)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]

    output_dir = sampling_params.get_output_dir(args.output_dir, args.model_name_or_path, args.benchmark_name)
    print(f"Saving results to: {output_dir}")

    # TODO: implement n_sampling 
    # TODO: batch inference and tokenization
    for example_idx, example in tqdm(enumerate(examples), total=len(examples)):
        # parse question and answer
        question = parse_question(example)
        messages = get_prompt(question, sampling_params.prompt_style, args.benchmark_name)

        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                do_sample = False if sampling_params.temperature==0 else True,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states

        num_tokens = len(hidden_states)
        num_layers = len(hidden_states[0])
        hidden_dim = hidden_states[0][0].size(-1)

        # don't include the embedding layer
        activations_tensor = torch.empty((num_tokens, num_layers-1, hidden_dim), dtype=hidden_states[0][0].dtype, device=hidden_states[0][0].device)

        # save hidden_states to a tensor
        for t in range(num_tokens):
            for l in range(1,num_layers):
                activations_tensor[t, l - 1] = hidden_states[t][l][0, -1]
        # Save activation tensor
        torch.save(activations_tensor.cpu(), os.path.join(output_dir, f"example_{example_idx}_activations.pt"))

        # Save output token ids
        output_token_ids = outputs.sequences[:,-num_tokens:]
        torch.save(output_token_ids.cpu(), os.path.join(output_dir, f"example_{example_idx}_output_ids.pt"))

        # Save output text (optional)
        output_text = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
        with open(os.path.join(output_dir, f"example_{example_idx}_output.txt"), "w") as f_txt:
            f_txt.write(output_text)

        # Save metadata to a JSONL file (append one line per example)
        metadata = {
            "example_idx": example_idx,
            "question": question,
            "output_text": output_text,
            "output_token_ids": output_token_ids.squeeze().tolist()
        }

        with open(os.path.join(output_dir, "metadata.jsonl"), "a") as f_meta:
            f_meta.write(json.dumps(metadata) + "\n")

if __name__ == "__main__":
    # Set seed before generation
    torch.manual_seed(sampling_params.seed)

    main()
