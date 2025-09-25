#!/bin/bash

# Scale experiment script for the final correct experiment
# This runs the full experiment with proper parameters

echo "🚀 Running Final Correct Subthought Length Experiment at Scale"
echo "============================================================="

# Create timestamp for results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./final_results_${TIMESTAMP}"

echo "📁 Results will be saved to: ${OUTPUT_DIR}"

# Run the full experiment with GPU optimization
python3 experiment_final_correct.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --traces_dir "/Users/atharvnaphade/Downloads/atharv/deepseek-qwen" \
  --output_dir "${OUTPUT_DIR}" \
  --num_examples 100 \
  --max_new_tokens 128 \
  --low_temp 0.0 \
  --high_temp 3.0 \
  --seed 42 \
  --device "auto"

echo ""
echo "🎉 Scale experiment completed!"
echo "📊 Check ${OUTPUT_DIR}/ for:"
echo "   - subthought_length_results.json (raw data)"
echo "   - subthought_length_boxplot.html (box plot)"
echo "   - subthought_length_scatter.html (scatter plot)"
echo "   - length_difference_histogram.html (difference histogram)"
