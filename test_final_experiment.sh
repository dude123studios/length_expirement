#!/bin/bash

# Test script for the final correct experiment
# This runs a small test to verify everything works

echo "🧪 Testing Final Correct Subthought Length Experiment"
echo "=================================================="

# Test with small parameters
python3 experiment_final_correct.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --traces_dir "/Users/atharvnaphade/Downloads/atharv/deepseek-qwen" \
  --output_dir "./test_final_results" \
  --num_examples 2 \
  --low_temp 0.0 \
  --high_temp 3.0 \
  --seed 42

echo ""
echo "✅ Test completed! Check ./test_final_results/ for results"
