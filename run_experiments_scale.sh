#!/bin/bash

# Scale-ready experiment scripts for subthought length analysis
# This script runs the temperature comparison experiment on both models

set -e  # Exit on any error

# Configuration
TEXT_BASE_DIR="/Users/atharvnaphade/Downloads/atharv"
DEEPSEEK_TEXT_DIR="${TEXT_BASE_DIR}/deepseek-qwen"
QWEN_TEXT_DIR="${TEXT_BASE_DIR}/qwen-instruct"
OUTPUT_BASE_DIR="./experiment_results_scale"
BENCHMARK_NAME="math"

# Model configurations
DEEPSEEK_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Experiment parameters
NUM_EXAMPLES=500  # Adjust based on your needs
HIGH_TEMP=1.2
MAX_NEW_TOKENS=128

echo "🚀 Starting Subthought Length Temperature Experiments"
echo "=================================================="
echo "DeepSeek model: ${DEEPSEEK_MODEL}"
echo "Qwen model: ${QWEN_MODEL}"
echo "Number of examples: ${NUM_EXAMPLES}"
echo "High temperature: ${HIGH_TEMP}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo ""

# Create output directories
mkdir -p "${OUTPUT_BASE_DIR}/deepseek"
mkdir -p "${OUTPUT_BASE_DIR}/qwen"

# Function to run experiment
run_experiment() {
    local model_name="$1"
    local text_dir="$2"
    local output_dir="$3"
    local model_display="$4"
    
    echo "🔬 Running experiment for ${model_display}..."
    echo "Text directory: ${text_dir}"
    echo "Output directory: ${output_dir}"
    echo ""
    
    python3 experiment_subthought_length_text.py \
        --model_name_or_path "${model_name}" \
        --benchmark_name "${BENCHMARK_NAME}" \
        --text_dir "${text_dir}" \
        --num_examples "${NUM_EXAMPLES}" \
        --high_temperature "${HIGH_TEMP}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --output_dir "${output_dir}"
    
    echo "✅ Completed experiment for ${model_display}"
    echo ""
}

# Check if text directories exist
if [ ! -d "${DEEPSEEK_TEXT_DIR}" ]; then
    echo "❌ Error: DeepSeek text directory not found: ${DEEPSEEK_TEXT_DIR}"
    exit 1
fi

if [ ! -d "${QWEN_TEXT_DIR}" ]; then
    echo "❌ Error: Qwen text directory not found: ${QWEN_TEXT_DIR}"
    exit 1
fi

# Run experiments
echo "Starting DeepSeek experiment..."
run_experiment "${DEEPSEEK_MODEL}" "${DEEPSEEK_TEXT_DIR}" "${OUTPUT_BASE_DIR}/deepseek" "DeepSeek-R1-Distill-Qwen-7B"

echo "Starting Qwen experiment..."
run_experiment "${QWEN_MODEL}" "${QWEN_TEXT_DIR}" "${OUTPUT_BASE_DIR}/qwen" "Qwen2.5-7B-Instruct"

echo "🎉 All experiments completed successfully!"
echo ""
echo "Results saved in:"
echo "- DeepSeek: ${OUTPUT_BASE_DIR}/deepseek/"
echo "- Qwen: ${OUTPUT_BASE_DIR}/qwen/"
echo ""
echo "📊 To view results, open the HTML files in each output directory:"
echo "- subthought_length_boxplot.html"
echo "- subthought_length_scatter.html"
echo "- length_difference_histogram.html"
echo "- complete_analysis.html"
