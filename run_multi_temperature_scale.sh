#!/bin/bash

# Scale script for multi-temperature experiment
# Runs with 100 examples and default temperatures [0.0, 0.7, 1.5, 2.5]

echo "🚀 Multi-Temperature Scale Experiment"
echo "===================================="
echo ""

# Check GPU availability
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ NVIDIA GPU detected"
else
    echo "⚠️  NVIDIA GPU not detected - will use CPU"
fi

# Check memory before starting
echo "💾 Checking system memory..."
free -h

# Check for data directory
DATA_DIR="./data/deepseek-qwen"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory not found: $DATA_DIR"
    echo ""
    echo "📁 Please upload your data to: $DATA_DIR"
    echo "   The directory should contain .txt files with reasoning traces"
    echo ""
    echo "💡 To upload data from your local machine:"
    echo "   scp -r /path/to/your/deepseek-qwen/ user@server:/path/to/experiment/data/"
    echo ""
    echo "💡 Or create the directory and upload files:"
    echo "   mkdir -p $DATA_DIR"
    echo "   # Then upload your .txt files to this directory"
    echo ""
    exit 1
fi

# Count data files
DATA_COUNT=$(ls -1 "$DATA_DIR"/*.txt 2>/dev/null | wc -l)
echo "✅ Data directory found: $DATA_COUNT text files"

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="multi_temp_scale_results_${TIMESTAMP}"

echo "📁 Output directory: $OUTPUT_DIR"
echo "🌡️  Temperatures: 0.0, 0.7, 1.5, 2.5"
echo "📊 Examples: 100"
echo ""

# Run the multi-temperature scale experiment
echo "🧪 Starting multi-temperature scale experiment..."
python3 experiment_multi_temperature.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --traces_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_examples 100 \
  --max_new_tokens 128 \
  --temperatures 0.0 0.7 1.5 2.5 \
  --seed 42 \
  --device "auto"

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Multi-temperature scale experiment completed successfully!"
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
    echo "❌ Multi-temperature scale experiment failed - check logs above for errors"
    exit 1
fi
