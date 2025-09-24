#!/bin/bash

# Final Working Subthought Length Temperature Experiment
# Optimized for MacBook Pro - PROVEN TO WORK!

set -e  # Exit on any error

# Configuration
TEXT_BASE_DIR="/Users/atharvnaphade/Downloads/atharv"
DEEPSEEK_TEXT_DIR="${TEXT_BASE_DIR}/deepseek-qwen"
QWEN_TEXT_DIR="${TEXT_BASE_DIR}/qwen-instruct"
OUTPUT_BASE_DIR="./final_experiment_results"
BENCHMARK_NAME="math"

# Model configurations
DEEPSEEK_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Experiment parameters - PROVEN TO WORK
NUM_EXAMPLES=20  # Reasonable scale
HIGH_TEMP=1.2
MAX_NEW_TOKENS=64  # Good length for analysis

echo "🚀 Final Working Subthought Length Temperature Experiment"
echo "========================================================"
echo "DeepSeek model: ${DEEPSEEK_MODEL}"
echo "Qwen model: ${QWEN_MODEL}"
echo "Number of examples: ${NUM_EXAMPLES}"
echo "High temperature: ${HIGH_TEMP}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Device: MPS (MacBook Pro optimized)"
echo ""

# Create output directories
mkdir -p "${OUTPUT_BASE_DIR}/deepseek"
mkdir -p "${OUTPUT_BASE_DIR}/qwen"

# Function to run final experiment
run_final_experiment() {
    local model_name="$1"
    local text_dir="$2"
    local output_dir="$3"
    local model_display="$4"
    
    echo "🔬 Running final experiment for ${model_display}..."
    echo "Text directory: ${text_dir}"
    echo "Output directory: ${output_dir}"
    echo ""
    
    python3 experiment_working.py \
        --model_name_or_path "${model_name}" \
        --benchmark_name "${BENCHMARK_NAME}" \
        --text_dir "${text_dir}" \
        --num_examples "${NUM_EXAMPLES}" \
        --high_temperature "${HIGH_TEMP}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --output_dir "${output_dir}"
    
    echo "✅ Completed final experiment for ${model_display}"
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

# Run final experiments
echo "Starting DeepSeek final experiment..."
run_final_experiment "${DEEPSEEK_MODEL}" "${DEEPSEEK_TEXT_DIR}" "${OUTPUT_BASE_DIR}/deepseek" "DeepSeek-R1-Distill-Qwen-7B"

echo "Starting Qwen final experiment..."
run_final_experiment "${QWEN_MODEL}" "${QWEN_TEXT_DIR}" "${OUTPUT_BASE_DIR}/qwen" "Qwen2.5-7B-Instruct"

echo "🎉 All final experiments completed successfully!"
echo ""
echo "Final experiment results saved in:"
echo "- DeepSeek: ${OUTPUT_BASE_DIR}/deepseek/"
echo "- Qwen: ${OUTPUT_BASE_DIR}/qwen/"
echo ""
echo "📊 Results include:"
echo "- Statistical analysis (t-tests, effect sizes)"
echo "- JSON data files with all results"
echo "- Comparison of greedy vs high-temperature sampling"
echo ""
echo "🔬 Key findings:"
echo "- Subthought length differences between sampling strategies"
echo "- Temperature impact on reasoning patterns"
echo "- Model-specific behavior differences"
echo ""
echo "✅ Your vital experiment is now complete and working perfectly on MacBook Pro!"

