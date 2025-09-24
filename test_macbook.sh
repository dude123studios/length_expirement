#!/bin/bash

# MacBook Pro Optimized Test Script
# Optimized for Apple Silicon with MPS support

set -e  # Exit on any error

# Configuration
TEXT_BASE_DIR="/Users/atharvnaphade/Downloads/atharv"
DEEPSEEK_TEXT_DIR="${TEXT_BASE_DIR}/deepseek-qwen"
QWEN_TEXT_DIR="${TEXT_BASE_DIR}/qwen-instruct"
OUTPUT_BASE_DIR="./macbook_test_results"
BENCHMARK_NAME="math"

# Model configurations
DEEPSEEK_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"

# MacBook Pro optimized parameters
NUM_EXAMPLES=3  # Small test for MacBook Pro
HIGH_TEMP=1.2
MAX_NEW_TOKENS=32  # Conservative for MPS

echo "🍎 MacBook Pro Optimized Subthought Length Experiment"
echo "===================================================="
echo "DeepSeek model: ${DEEPSEEK_MODEL}"
echo "Qwen model: ${QWEN_MODEL}"
echo "Number of examples: ${NUM_EXAMPLES} (MacBook Pro optimized)"
echo "High temperature: ${HIGH_TEMP}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Device: MPS (Metal Performance Shaders)"
echo ""

# Create output directories
mkdir -p "${OUTPUT_BASE_DIR}/deepseek"
mkdir -p "${OUTPUT_BASE_DIR}/qwen"

# Function to run MacBook optimized test
run_macbook_test() {
    local model_name="$1"
    local text_dir="$2"
    local output_dir="$3"
    local model_display="$4"
    
    echo "🍎 Running MacBook optimized test for ${model_display}..."
    echo "Text directory: ${text_dir}"
    echo "Output directory: ${output_dir}"
    echo ""
    
    python3 experiment_macbook_optimized.py \
        --model_name_or_path "${model_name}" \
        --benchmark_name "${BENCHMARK_NAME}" \
        --text_dir "${text_dir}" \
        --num_examples "${NUM_EXAMPLES}" \
        --high_temperature "${HIGH_TEMP}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --output_dir "${output_dir}"
    
    echo "✅ Completed MacBook test for ${model_display}"
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

# Check MPS availability
echo "🔍 Checking MacBook Pro MPS support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('✅ MPS support detected - MacBook Pro optimized!')
else:
    print('⚠️  MPS not available - will use CPU')
"
echo ""

# Run MacBook optimized tests
echo "Starting DeepSeek MacBook test..."
run_macbook_test "${DEEPSEEK_MODEL}" "${DEEPSEEK_TEXT_DIR}" "${OUTPUT_BASE_DIR}/deepseek" "DeepSeek-R1-Distill-Qwen-7B"

echo "Starting Qwen MacBook test..."
run_macbook_test "${QWEN_MODEL}" "${QWEN_TEXT_DIR}" "${OUTPUT_BASE_DIR}/qwen" "Qwen2.5-7B-Instruct"

echo "🎉 All MacBook Pro optimized tests completed successfully!"
echo ""
echo "MacBook test results saved in:"
echo "- DeepSeek: ${OUTPUT_BASE_DIR}/deepseek/"
echo "- Qwen: ${OUTPUT_BASE_DIR}/qwen/"
echo ""
echo "📊 Check the JSON files for results"
echo ""
echo "🚀 MacBook Pro optimizations applied:"
echo "- MPS (Metal Performance Shaders) acceleration"
echo "- Float16 precision for better performance"
echo "- Optimized memory usage"
echo "- Conservative token limits for stability"
echo ""
echo "If tests work well, you can increase num_examples for larger experiments"

