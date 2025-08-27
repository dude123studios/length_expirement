#!/bin/bash

uv run plot_activations_comparison.py \
  --model_name_or_path1 "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --model_name_or_path2 "Qwen/Qwen2.5-7B-Instruct" \
  --activations_dir1 "/network/scratch/p/prateek.humane/llm-reasoning-activations/activations/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B/math/temp0.0_top-p1.0" \
  --activations_dir2 "/network/scratch/p/prateek.humane/llm-reasoning-activations/activations/Qwen_Qwen2.5-7B-Instruct/math/temp0.0_top-p1.0" \
  --plot_dir "/home/mila/p/prateek.humane/scratch/llm-reasoning-activations/plots/DeepSeek-R1-Distill-Qwen-7B_vs_Qwen-7b-Instruct"

uv run plot_activations_interactive.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --activations_dir "/network/scratch/p/prateek.humane/llm-reasoning-activations/activations/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B/math/temp0.0_top-p1.0" \
  --end_idx 500

uv run plot_activations_interactive.py \
  --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
  --activations_dir "/network/scratch/p/prateek.humane/llm-reasoning-activations/activations/Qwen_Qwen2.5-7B-Instruct/math/temp0.0_top-p1.0" \
  --end_idx 500

uv run plot_dynamics.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --activations_dir "/network/scratch/p/prateek.humane/llm-reasoning-activations/activations/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B/math/temp0.0_top-p1.0" \
  --plot_dir "/home/mila/p/prateek.humane/scratch/llm-reasoning-activations/plots/DeepSeek-R1-Distill-Qwen-7B/dynamics" \
  --end_idx 1
