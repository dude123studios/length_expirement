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

# plot deepseek pca
uv run plot_dynamics.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --activations_dir "/network/scratch/p/prateek.humane/llm-reasoning-activations/activations/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B/math/temp0.0_top-p1.0" \
  --plot_dir "/home/mila/p/prateek.humane/scratch/llm-reasoning-activations/plots/DeepSeek-R1-Distill-Qwen-7B/dynamics" \
  --end_idx 1

# plot deepseek momentum
uv run plot_activations_correlation.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --activations_dir "/network/scratch/p/prateek.humane/llm-reasoning-activations/activations/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B/math/temp0.0_top-p1.0" \
  --plot_dir "/home/mila/p/prateek.humane/scratch/llm-reasoning-activations/plots/DeepSeek-R1-Distill-Qwen-7B/comparison_matrix/" \
  --end_idx 5 \
  --do_pca_momentum True --do_momentum True

# plot qwen instruct momentum
uv run plot_activations_correlation.py \
  --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
  --activations_dir "/network/scratch/p/prateek.humane/llm-reasoning-activations/activations/Qwen_Qwen2.5-7B-Instruct/math/temp0.0_top-p1.0" \
  --plot_dir "/home/mila/p/prateek.humane/scratch/llm-reasoning-activations/plots/Qwen_Qwen2.5-7B-Instruct/comparison_matrix/" \
  --end_idx 5 \
  --do_pca_momentum True --do_momentum True

uv run plot_prob_correct.py \
  --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
  --activations_dir "/network/scratch/p/prateek.humane/llm-reasoning-activations/activations/Qwen_Qwen2.5-7B-Instruct/math/temp0.0_top-p1.0"

uv run plot_prob_correct.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --activations_dir "/network/scratch/p/prateek.humane/llm-reasoning-activations/activations/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B/math/temp0.0_top-p1.0"
