#!/bin/bash

# Test experiment scripts for subthought length analysis
# This script runs small-scale tests to validate the setup

set -e  # Exit on any error

# Configuration
TEXT_BASE_DIR="/Users/atharvnaphade/Downloads/atharv"
DEEPSEEK_TEXT_DIR="${TEXT_BASE_DIR}/deepseek-qwen"
QWEN_TEXT_DIR="${TEXT_BASE_DIR}/qwen-instruct"
OUTPUT_BASE_DIR="./test_results"
BENCHMARK_NAME="math"

# Model configurations
DEEPSEEK_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Test parameters (small scale)
NUM_EXAMPLES=5  # Small test
HIGH_TEMP=1.2
MAX_NEW_TOKENS=64  # Smaller for faster testing

echo "🧪 Running Test Experiments for Subthought Length Analysis"
echo "========================================================="
echo "DeepSeek model: ${DEEPSEEK_MODEL}"
echo "Qwen model: ${QWEN_MODEL}"
echo "Number of examples: ${NUM_EXAMPLES} (TEST MODE)"
echo "High temperature: ${HIGH_TEMP}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo ""

# Create output directories
mkdir -p "${OUTPUT_BASE_DIR}/deepseek"
mkdir -p "${OUTPUT_BASE_DIR}/qwen"

# Function to run test experiment
run_test_experiment() {
    local model_name="$1"
    local text_dir="$2"
    local output_dir="$3"
    local model_display="$4"
    
    echo "🔬 Running test experiment for ${model_display}..."
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
    
    echo "✅ Completed test experiment for ${model_display}"
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

# Run test experiments
echo "Starting DeepSeek test experiment..."
run_test_experiment "${DEEPSEEK_MODEL}" "${DEEPSEEK_TEXT_DIR}" "${OUTPUT_BASE_DIR}/deepseek" "DeepSeek-R1-Distill-Qwen-7B"

echo "Starting Qwen test experiment..."
run_test_experiment "${QWEN_MODEL}" "${QWEN_TEXT_DIR}" "${OUTPUT_BASE_DIR}/qwen" "Qwen2.5-7B-Instruct"

echo "🎉 All test experiments completed successfully!"
echo ""
echo "Test results saved in:"
echo "- DeepSeek: ${OUTPUT_BASE_DIR}/deepseek/"
echo "- Qwen: ${OUTPUT_BASE_DIR}/qwen/"
echo ""
echo "📊 To view test results, open the HTML files in each output directory"
echo ""
echo "🚀 If tests look good, run the full-scale experiments with:"
echo "   ./run_experiments_scale.sh"
