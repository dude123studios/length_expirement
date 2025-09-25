#!/bin/bash

# Script to run the test experiment 4 times with memory efficiency
# This ensures we get robust results while managing memory usage

echo "🧪 Running Test Experiment 4 Times (Memory Optimized)"
echo "====================================================="

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="./test_results_4_runs_${TIMESTAMP}"

echo "📁 Results will be saved to: ${BASE_OUTPUT_DIR}"
mkdir -p "${BASE_OUTPUT_DIR}"

# Function to run a single test experiment
run_single_test() {
    local run_number=$1
    local output_dir="${BASE_OUTPUT_DIR}/run_${run_number}"
    
    echo ""
    echo "🔄 Running Test Experiment ${run_number}/4"
    echo "=========================================="
    
    # Run the experiment with memory optimizations
    python3 experiment_final_correct.py \
      --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
      --traces_dir "/Users/atharvnaphade/Downloads/atharv/deepseek-qwen" \
      --output_dir "${output_dir}" \
      --num_examples 2 \
      --max_new_tokens 128 \
      --low_temp 0.0 \
      --high_temp 3.0 \
      --seed $((42 + run_number)) \
      --device "auto"
    
    # Memory cleanup between runs
    echo "🧹 Cleaning up memory..."
    if command -v python3 &> /dev/null; then
        python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('Memory cleared')"
    fi
    
    # Small delay to ensure cleanup
    sleep 2
    
    echo "✅ Run ${run_number} completed"
}

# Run all 4 experiments
for i in {1..4}; do
    run_single_test $i
done

echo ""
echo "🎉 All 4 test experiments completed!"
echo "📊 Results saved in: ${BASE_OUTPUT_DIR}/"
echo ""
echo "📁 Directory structure:"
echo "   ${BASE_OUTPUT_DIR}/"
echo "   ├── run_1/"
echo "   ├── run_2/"
echo "   ├── run_3/"
echo "   └── run_4/"
echo ""
echo "📈 Each run contains:"
echo "   - subthought_length_results.json"
echo "   - length_distribution_histogram.html"
echo "   - length_difference_histogram.html"
echo "   - length_analysis_combined.html"
echo "   - publication-ready PDF plots"
echo ""
echo "🔍 To analyze results across all runs:"
echo "   python3 analyze_4_runs.py --results_dir ${BASE_OUTPUT_DIR}"
