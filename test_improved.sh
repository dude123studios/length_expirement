#!/bin/bash

# Improved Subthought Length Experiment
# - Longer sequences (128 tokens)
# - Higher temperature (2.5) for more diversity
# - More examples (10)
# - Starting from different positions in reasoning traces

set -e  # Exit on any error

# Configuration
TEXT_BASE_DIR="/Users/atharvnaphade/Downloads/atharv"
QWEN_TEXT_DIR="${TEXT_BASE_DIR}/qwen-instruct"
OUTPUT_DIR="./improved_results"
BENCHMARK_NAME="math"

# Model configuration
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Improved parameters
NUM_EXAMPLES=10  # More examples
HIGH_TEMP=2.5    # Higher temperature for more diversity
MAX_NEW_TOKENS=128  # Much longer sequences

echo "🚀 Improved Subthought Length Experiment"
echo "======================================="
echo "Model: ${QWEN_MODEL}"
echo "Number of examples: ${NUM_EXAMPLES}"
echo "High temperature: ${HIGH_TEMP} (higher for more diversity)"
echo "Max new tokens: ${MAX_NEW_TOKENS} (longer sequences)"
echo "Starting positions: Different parts of reasoning traces"
echo ""

# Check if text directory exists
if [ ! -d "${QWEN_TEXT_DIR}" ]; then
    echo "❌ Error: Qwen text directory not found: ${QWEN_TEXT_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "🔬 Running improved experiment..."
echo "Text directory: ${QWEN_TEXT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Run the improved experiment
python3 experiment_working.py \
    --model_name_or_path "${QWEN_MODEL}" \
    --benchmark_name "${BENCHMARK_NAME}" \
    --text_dir "${QWEN_TEXT_DIR}" \
    --num_examples "${NUM_EXAMPLES}" \
    --high_temperature "${HIGH_TEMP}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --output_dir "${OUTPUT_DIR}"

echo "✅ Improved experiment completed!"
echo ""
echo "Results saved in: ${OUTPUT_DIR}"
echo ""
echo "📊 This should show:"
echo "- Longer subthought sequences"
echo "- More diverse high-temperature outputs"
echo "- Better comparison between greedy vs high-temp"
echo "- Multiple decision token positions"
echo ""
echo "🚀 If this works well, we can scale up to 50+ examples for the full experiment"

