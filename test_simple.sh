#!/bin/bash

# Simple test script for the subthought length experiment
# This script runs a minimal test to validate the setup

set -e  # Exit on any error

# Configuration
TEXT_BASE_DIR="/Users/atharvnaphade/Downloads/atharv"
DEEPSEEK_TEXT_DIR="${TEXT_BASE_DIR}/deepseek-qwen"
QWEN_TEXT_DIR="${TEXT_BASE_DIR}/qwen-instruct"
OUTPUT_BASE_DIR="./simple_test_results"
BENCHMARK_NAME="math"

# Model configurations
DEEPSEEK_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Test parameters (very small)
NUM_EXAMPLES=2  # Very small test
HIGH_TEMP=1.2
MAX_NEW_TOKENS=32  # Small for faster testing

echo "🧪 Running Simple Test for Subthought Length Analysis"
echo "===================================================="
echo "DeepSeek model: ${DEEPSEEK_MODEL}"
echo "Qwen model: ${QWEN_MODEL}"
echo "Number of examples: ${NUM_EXAMPLES} (MINIMAL TEST)"
echo "High temperature: ${HIGH_TEMP}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo ""

# Create output directories
mkdir -p "${OUTPUT_BASE_DIR}/deepseek"
mkdir -p "${OUTPUT_BASE_DIR}/qwen"

# Function to run simple test
run_simple_test() {
    local model_name="$1"
    local text_dir="$2"
    local output_dir="$3"
    local model_display="$4"
    
    echo "🔬 Running simple test for ${model_display}..."
    echo "Text directory: ${text_dir}"
    echo "Output directory: ${output_dir}"
    echo ""
    
    python3 experiment_simple.py \
        --model_name_or_path "${model_name}" \
        --benchmark_name "${BENCHMARK_NAME}" \
        --text_dir "${text_dir}" \
        --num_examples "${NUM_EXAMPLES}" \
        --high_temperature "${HIGH_TEMP}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --output_dir "${output_dir}"
    
    echo "✅ Completed simple test for ${model_display}"
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

# Run simple tests
echo "Starting DeepSeek simple test..."
run_simple_test "${DEEPSEEK_MODEL}" "${DEEPSEEK_TEXT_DIR}" "${OUTPUT_BASE_DIR}/deepseek" "DeepSeek-R1-Distill-Qwen-7B"

echo "Starting Qwen simple test..."
run_simple_test "${QWEN_MODEL}" "${QWEN_TEXT_DIR}" "${OUTPUT_BASE_DIR}/qwen" "Qwen2.5-7B-Instruct"

echo "🎉 All simple tests completed successfully!"
echo ""
echo "Simple test results saved in:"
echo "- DeepSeek: ${OUTPUT_BASE_DIR}/deepseek/"
echo "- Qwen: ${OUTPUT_BASE_DIR}/qwen/"
echo ""
echo "📊 Check the JSON files for results"
echo ""
echo "🚀 If simple tests work, you can increase num_examples and run larger experiments"

