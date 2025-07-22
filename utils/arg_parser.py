import os
import yaml
from dataclasses import dataclass, fields
from argparse import Namespace
from typing import Optional, Union
from pathlib import Path

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

@dataclass
class SamplingParams:
    prompt_style: str = "qwen-instruct"
    temperature: float = 0.6
    top_p: float = 0.95
    k: int = 1
    n_sampling: int = 1
    max_tokens: int = 16384
    seed: int = 42

    @classmethod
    def from_args_and_yaml(cls, args: Namespace, yaml_path_or_name: Optional[str] = None) -> "SamplingParams":
        yaml_config = {}
        if yaml_path_or_name:
            if not os.path.exists(yaml_path_or_name):
                yaml_path = os.path.join("configs", yaml_path_or_name)
            else:
                yaml_path = yaml_path_or_name
            yaml_config = load_yaml(yaml_path)
        config = {}

        for field in fields(cls):
            key = field.name
            val = getattr(args, key, None)
            config[key] = val if val is not None else yaml_config.get(key, field.default)

        return cls(**config)

    def get_output_dir(self, base_output_dir: Path, model_name_or_path: Union[str, Path], benchmark_name: str) -> Path:
        # Create a unique identifier for the sampling params
        param_id = f"temp{self.temperature}_top-p{self.top_p}"

        print(model_name_or_path)

        if os.path.exists(model_name_or_path):
            model_id = os.path.basename(os.path.normpath(model_name_or_path))
        else:
            model_id = model_name_or_path.replace("/", "_")

        print(f"Model ID: {model_id}, Benchmark: {benchmark_name}, Params: {param_id}")
        # Define structured subdirectory: output_dir/model_id/param_id/
        output_subdir = os.path.join(base_output_dir, model_id, benchmark_name, param_id)
        os.makedirs(output_subdir, exist_ok=True)
        return output_subdir

    # def get_output_dir(self, model_name_or_path: Union[str, Path], base_output_dir: Path, benchmark_name: str) -> Path:
    #     # Ensure model_name_or_path is a Path
    #     model_path = Path(model_name_or_path)
    #
    #     param_id = f"temp{self.temperature}_top-p"
    #
    #     # If it's a real path, extract the basename; else sanitize model name
    #     if model_path.exists():
    #         model_id = model_path.name
    #     else:
    #         model_id = str(model_name_or_path).replace("/", "_")
    #
    #     # Final output directory
    #     output_subdir = base_output_dir / model_id / benchmark_name / param_id
    #     output_subdir.mkdir(parents=True, exist_ok=True)
    #
    #     return output_subdir
