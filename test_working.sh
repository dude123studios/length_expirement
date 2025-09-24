#!/bin/bash

# Working Test Script - Uses proven patterns from existing working code

set -e  # Exit on any error

# Configuration
TEXT_BASE_DIR="/Users/atharvnaphade/Downloads/atharv"
QWEN_TEXT_DIR="${TEXT_BASE_DIR}/qwen-instruct"
OUTPUT_DIR="./working_test_results"
BENCHMARK_NAME="math"

# Model configuration
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Test parameters
NUM_EXAMPLES=2  # Very small test
HIGH_TEMP=1.2
MAX_NEW_TOKENS=16  # Very small for testing

echo "🚀 Working Subthought Length Experiment Test"
echo "==========================================="
echo "Model: ${QWEN_MODEL}"
echo "Number of examples: ${NUM_EXAMPLES} (MINIMAL TEST)"
echo "High temperature: ${HIGH_TEMP}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo ""

# Check if text directory exists
if [ ! -d "${QWEN_TEXT_DIR}" ]; then
    echo "❌ Error: Qwen text directory not found: ${QWEN_TEXT_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "🔬 Running working experiment test..."
echo "Text directory: ${QWEN_TEXT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Run the working experiment
python3 experiment_working.py \
    --model_name_or_path "${QWEN_MODEL}" \
    --benchmark_name "${BENCHMARK_NAME}" \
    --text_dir "${QWEN_TEXT_DIR}" \
    --num_examples "${NUM_EXAMPLES}" \
    --high_temperature "${HIGH_TEMP}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --output_dir "${OUTPUT_DIR}"

echo "✅ Working experiment test completed!"
echo ""
echo "Results saved in: ${OUTPUT_DIR}"
echo ""
echo "📊 Check the JSON file for results"
echo ""
echo "🚀 If this works, you can increase num_examples and max_new_tokens for larger experiments"

