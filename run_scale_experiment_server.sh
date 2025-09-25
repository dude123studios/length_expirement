#!/bin/bash

# Scale experiment script for GPU server deployment
# This runs the full experiment with comprehensive analysis and visualizations

echo "🚀 Running Scale Subthought Length Experiment on GPU Server"
echo "=========================================================="

# Create timestamp for results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./scale_results_${TIMESTAMP}"

echo "📁 Results will be saved to: ${OUTPUT_DIR}"
echo "🖥️  GPU Server Configuration:"
echo "   - Model: DeepSeek-R1-Distill-Qwen-7B"
echo "   - Examples: 500"
echo "   - Max tokens: 128"
echo "   - Regular temp: 0.0 (greedy)"
echo "   - High temp: 3.0"
echo "   - Stopping: Highest probability decision token"
echo ""

# Check GPU availability
echo "🔍 Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.device_count()} GPU(s)')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')
else:
    print('❌ CUDA not available - falling back to CPU')
"

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

# Run the full scale experiment
echo ""
echo "🧪 Starting scale experiment..."
python3 experiment_final_correct.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --traces_dir "$DATA_DIR" \
  --output_dir "${OUTPUT_DIR}" \
  --num_examples 500 \
  --max_new_tokens 128 \
  --low_temp 0.0 \
  --high_temp 3.0 \
  --seed 42 \
  --device "auto"

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Scale experiment completed successfully!"
    echo ""
    echo "📊 Generated Analysis Files:"
    echo "   📈 Essential Results:"
    echo "      - subthought_length_results.json (raw data + statistics)"
    echo "      - scale_experiment_aggregated_results.html (4-panel aggregated analysis)"
    echo "      - scale_experiment_summary.html (summary statistics)"
    echo ""
    echo "   📊 CSV Data (PARAMOUNT):"
    echo "      - scale_experiment_results.csv (greedy_length, high_temp_length, difference)"
    echo ""
    echo "📁 Results directory: ${OUTPUT_DIR}/"
    echo ""
    echo "🔍 Key Statistics to Check:"
    echo "   - Mean difference (Regular - High): Check if positive/negative"
    echo "   - T-test p-value: Statistical significance"
    echo "   - Effect size: Practical significance"
    echo "   - IQR differences: Distribution spread"
    echo ""
    echo "📈 Expected Results:"
    echo "   - If high temp → shorter reasoning: Positive differences"
    echo "   - If high temp → longer reasoning: Negative differences"
    echo "   - Statistical significance: p < 0.05"
    echo ""
    echo "✅ Ready for publication analysis!"
else
    echo "❌ Experiment failed - check logs above for errors"
    exit 1
fi
