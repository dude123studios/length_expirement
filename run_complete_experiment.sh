#!/bin/bash

# Complete Subthought Length Temperature Experiment Runner
# This script handles everything from environment setup to running the full experiment
# Usage: ./run_complete_experiment.sh [test|scale]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

# Get experiment type (test or scale)
EXPERIMENT_TYPE=${1:-"test"}
if [[ "$EXPERIMENT_TYPE" != "test" && "$EXPERIMENT_TYPE" != "scale" ]]; then
    print_error "Invalid experiment type. Use 'test' or 'scale'"
    echo "Usage: $0 [test|scale]"
    echo "  test  - Run with 2 examples (5-10 minutes)"
    echo "  scale - Run with 100 examples (45-90 minutes)"
    exit 1
fi

print_header "🚀 Complete Subthought Length Temperature Experiment"
print_status "Experiment type: $EXPERIMENT_TYPE"

# Step 1: Check Python and create virtual environment
print_header "📋 Step 1: Environment Setup"

print_status "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
print_success "Python version: $PYTHON_VERSION"

# Check if we're already in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_success "Virtual environment detected: $VIRTUAL_ENV"
    VENV_ACTIVATED=true
else
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
    VENV_ACTIVATED=false
fi

# Step 2: Install dependencies
print_header "📦 Step 2: Installing Dependencies"

print_status "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

print_status "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt > /dev/null 2>&1
    print_success "Requirements installed successfully"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Step 3: Verify installation
print_header "🔍 Step 3: Verifying Installation"

print_status "Checking key packages..."
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__}')" 2>/dev/null || { print_error "PyTorch not installed"; exit 1; }
python3 -c "import transformers; print(f'✅ Transformers {transformers.__version__}')" 2>/dev/null || { print_error "Transformers not installed"; exit 1; }
python3 -c "import plotly; print(f'✅ Plotly {plotly.__version__}')" 2>/dev/null || { print_error "Plotly not installed"; exit 1; }
python3 -c "import pandas; print(f'✅ Pandas {pandas.__version__}')" 2>/dev/null || { print_error "Pandas not installed"; exit 1; }
python3 -c "import numpy; print(f'✅ NumPy {numpy.__version__}')" 2>/dev/null || { print_error "NumPy not installed"; exit 1; }
python3 -c "import scipy; print(f'✅ SciPy {scipy.__version__}')" 2>/dev/null || { print_error "SciPy not installed"; exit 1; }
python3 -c "import seaborn; print(f'✅ Seaborn {seaborn.__version__}')" 2>/dev/null || { print_error "Seaborn not installed"; exit 1; }
python3 -c "import matplotlib; print(f'✅ Matplotlib {matplotlib.__version__}')" 2>/dev/null || { print_error "Matplotlib not installed"; exit 1; }

# Check CUDA availability
print_status "Checking CUDA availability..."
CUDA_INFO=$(python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null)
print_success "$CUDA_INFO"

# Step 4: Check data directories
print_header "📁 Step 4: Checking Data Directories"

TEXT_BASE_DIR="/Users/atharvnaphade/Downloads/atharv"
DEEPSEEK_DIR="${TEXT_BASE_DIR}/deepseek-qwen"

if [ -d "${DEEPSEEK_DIR}" ]; then
    DEEPSEEK_COUNT=$(ls -1 "${DEEPSEEK_DIR}"/*.txt 2>/dev/null | wc -l)
    print_success "DeepSeek directory found: ${DEEPSEEK_COUNT} text files"
    
    if [ "$DEEPSEEK_COUNT" -lt 2 ]; then
        print_warning "Only $DEEPSEEK_COUNT files found. You need at least 2 for the test experiment."
    fi
    
    if [ "$EXPERIMENT_TYPE" = "scale" ] && [ "$DEEPSEEK_COUNT" -lt 100 ]; then
        print_warning "Only $DEEPSEEK_COUNT files found. Scale experiment needs 100+ files for best results."
    fi
else
    print_error "DeepSeek directory not found: ${DEEPSEEK_DIR}"
    print_error "Please ensure your trace files are in the correct location."
    exit 1
fi

# Step 5: Make scripts executable
print_header "🔧 Step 5: Preparing Scripts"

print_status "Making scripts executable..."
chmod +x test_final_experiment.sh 2>/dev/null || true
chmod +x run_final_experiment_scale.sh 2>/dev/null || true
chmod +x generate_publication_plots.py 2>/dev/null || true
print_success "Scripts prepared"

# Step 6: Run the experiment
print_header "🧪 Step 6: Running Experiment"

if [ "$EXPERIMENT_TYPE" = "test" ]; then
    print_status "Running TEST experiment (2 examples, ~5-10 minutes)..."
    print_status "Parameters: T=0.7 vs T=2.5, max_tokens=128"
    
    # Run test experiment
    python3 experiment_final_correct.py \
        --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --traces_dir "/Users/atharvnaphade/Downloads/atharv/deepseek-qwen" \
        --output_dir "./test_final_results" \
        --num_examples 2 \
        --max_new_tokens 128 \
        --low_temp 0.7 \
        --high_temp 2.5 \
        --seed 42 \
        --device "auto"
    
    RESULT_DIR="./test_final_results"
    
elif [ "$EXPERIMENT_TYPE" = "scale" ]; then
    print_status "Running SCALE experiment (100 examples, ~45-90 minutes)..."
    print_status "Parameters: T=0.7 vs T=2.5, max_tokens=128"
    
    # Create timestamp for results
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="./final_results_${TIMESTAMP}"
    
    # Run scale experiment
    python3 experiment_final_correct.py \
        --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --traces_dir "/Users/atharvnaphade/Downloads/atharv/deepseek-qwen" \
        --output_dir "${OUTPUT_DIR}" \
        --num_examples 100 \
        --max_new_tokens 128 \
        --low_temp 0.7 \
        --high_temp 2.5 \
        --seed 42 \
        --device "auto"
    
    RESULT_DIR="${OUTPUT_DIR}"
fi

# Step 7: Check results
print_header "📊 Step 7: Results Summary"

if [ -f "${RESULT_DIR}/subthought_length_results.json" ]; then
    print_success "Experiment completed successfully!"
    
    # Extract key statistics
    if command -v jq &> /dev/null; then
        REGULAR_AVG=$(jq -r '.summary.regular_avg' "${RESULT_DIR}/subthought_length_results.json")
        HIGH_TEMP_AVG=$(jq -r '.summary.high_temp_avg' "${RESULT_DIR}/subthought_length_results.json")
        P_VALUE=$(jq -r '.summary.p_value' "${RESULT_DIR}/subthought_length_results.json")
        SIGNIFICANT=$(jq -r '.summary.significant' "${RESULT_DIR}/subthought_length_results.json")
        
        print_success "Results:"
        echo "  Regular temp (T=0.7): ${REGULAR_AVG} tokens"
        echo "  High temp (T=2.5): ${HIGH_TEMP_AVG} tokens"
        echo "  Difference: $(echo "$HIGH_TEMP_AVG - $REGULAR_AVG" | bc -l) tokens"
        echo "  P-value: ${P_VALUE}"
        echo "  Significant: ${SIGNIFICANT}"
    else
        print_success "Results saved to ${RESULT_DIR}/subthought_length_results.json"
    fi
    
    # List generated files
    print_success "Generated files:"
    ls -la "${RESULT_DIR}"/*.pdf 2>/dev/null | while read line; do
        echo "  📄 $(basename $(echo $line | awk '{print $NF}'))"
    done
    ls -la "${RESULT_DIR}"/*.json 2>/dev/null | while read line; do
        echo "  📊 $(basename $(echo $line | awk '{print $NF}'))"
    done
    
else
    print_error "Experiment failed - no results file found"
    exit 1
fi

# Step 8: Final instructions
print_header "🎉 Step 8: Complete!"

print_success "Experiment completed successfully!"
echo ""
print_status "📁 Results location: ${RESULT_DIR}"
print_status "📊 Main results file: ${RESULT_DIR}/subthought_length_results.json"
print_status "📄 Publication plots: ${RESULT_DIR}/*_publication.pdf"
echo ""
print_status "📋 Next steps:"
echo "  1. Review the results in ${RESULT_DIR}/"
echo "  2. Open the PDF files to see publication-ready plots"
echo "  3. Check the JSON file for detailed statistics"
echo "  4. If running test, consider running scale experiment: $0 scale"
echo ""

if [ "$VENV_ACTIVATED" = "false" ]; then
    print_warning "Remember: Virtual environment is still active"
    print_warning "To deactivate: deactivate"
fi

print_success "🎯 All done! Your subthought length experiment is complete."
