import os
import json
import random
from datasets import load_dataset, Dataset, concatenate_datasets
import importlib.util
from typing import Iterable, Union, Any
from pathlib import Path


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def load_benchmark(data_name):
    data_dir = './benchmarks'
    split = 'test'
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        if data_name == "math":
            dataset = load_dataset("competition_math", split=split, name="main", cache_dir=f"{data_dir}/temp")
        elif data_name == "theorem-qa":
            dataset = load_dataset("wenhu/TheoremQA", split=split)
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split)
        elif data_name == "gsm-hard":
            dataset = load_dataset("reasoning-machines/gsm-hard", split="train")
        elif data_name == "svamp":
            # evaluate on training set + test set 
            dataset = load_dataset("ChilleD/SVAMP", split="train")
            dataset = concatenate_datasets([dataset, load_dataset("ChilleD/SVAMP", split="test")])
        elif data_name == "asdiv":
            dataset = load_dataset("EleutherAI/asdiv", split="validation")
            dataset = dataset.filter(lambda x: ";" not in x['answer']) # remove multi-answer examples
        elif data_name == "mawps":
            examples = []
            # four sub-tasks
            for data_name in ["singleeq", "singleop", "addsub", "multiarith"]:
                sub_examples = list(load_jsonl(f"{data_dir}/mawps/{data_name}.jsonl"))
                for example in sub_examples:
                    example['type'] = data_name
                examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "finqa":
            dataset = load_dataset("dreamerdeo/finqa", split=split, name="main")
            dataset = dataset.select(random.sample(range(len(dataset)), 1000))
        elif data_name == "tabmwp":
            examples = []
            with open(f"{data_dir}/tabmwp/tabmwp_{split}.json", "r") as f:
                data_dict = json.load(f)
                examples.extend(data_dict.values())
            dataset = Dataset.from_list(examples)
            dataset = dataset.select(random.sample(range(len(dataset)), 1000))
        elif data_name == "bbh":
            examples = []
            for data_name in ["reasoning_about_colored_objects", "penguins_in_a_table",\
                            "date_understanding", "repeat_copy_logic", "object_counting"]:
                with open(f"{data_dir}/bbh/bbh/{data_name}.json", "r") as f:
                    sub_examples = json.load(f)["examples"]
                    for example in sub_examples:
                        example['type'] = data_name
                    examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        else:
            raise NotImplementedError(data_name)

        examples = list(dataset)
        examples = [lower_keys(example) for example in examples]
        dataset = Dataset.from_list(examples)
        os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
        dataset.to_json(data_file)

    # add 'idx' in the first column
    if 'idx' not in examples[0]:
        examples = [{'idx': i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x['idx'])
    return examples

def parse_question(example):
    question = ""
    for key in ["question", "problem", "Question", "input"]:
        if key in example:
            question = example[key]
            break
    
    return question.strip()

def get_prompt(question, prompt_type, data_name):
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")

    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")

    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    cur_prompt = few_shot_prompt + question_format.format(question=question) if few_shot_prompt else question_format.format(question=question)
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": cur_prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": cur_prompt}
        ]

    return messages 
