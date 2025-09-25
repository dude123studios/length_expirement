#!/bin/bash

# Test script for multi-temperature experiment
# Runs with 2 examples and default temperatures [0.0, 0.7, 1.5, 2.5]

echo "🧪 Testing Multi-Temperature Experiment"
echo "======================================"
echo ""

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="test_multi_temp_results_${TIMESTAMP}"

echo "📁 Output directory: $OUTPUT_DIR"
echo "🌡️  Temperatures: 0.0, 0.7, 1.5, 2.5"
echo "📊 Examples: 2"
echo ""

# Run the multi-temperature experiment
python3 experiment_multi_temperature.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --traces_dir "./data/deepseek-qwen" \
  --output_dir "$OUTPUT_DIR" \
  --num_examples 2 \
  --max_new_tokens 128 \
  --temperatures 0.0 0.7 1.5 2.5 \
  --seed 42 \
  --device "auto"

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Multi-temperature test experiment completed successfully!"
    echo ""
    echo "📊 Results generated:"
    echo "   - multi_temperature_results.csv (PARAMOUNT - your data)"
    echo "   - multi_temperature_analysis.html (4-panel analysis)"
    echo "   - multi_temperature_results.json (raw data)"
    echo ""
    echo "📁 Check the $OUTPUT_DIR/ directory for all files"
    echo ""
    echo "✅ Ready for analysis!"
else
    echo "❌ Multi-temperature test experiment failed - check logs above for errors"
    exit 1
fi
