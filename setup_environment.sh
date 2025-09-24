#!/bin/bash

# Setup script for the subthought length temperature experiment
# This script ensures all dependencies are installed correctly

set -e  # Exit on any error

echo "🚀 Setting up environment for Subthought Length Temperature Experiment"
echo "=================================================================="

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo ""
fi

# Install requirements
echo "📦 Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    echo "✅ Requirements installed successfully"
else
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

# Check if key packages are available
echo "🔍 Verifying key packages..."
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
python3 -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"
python3 -c "import plotly; print(f'✅ Plotly {plotly.__version__}')"
python3 -c "import pandas; print(f'✅ Pandas {pandas.__version__}')"
python3 -c "import numpy; print(f'✅ NumPy {numpy.__version__}')"
python3 -c "import scipy; print(f'✅ SciPy {scipy.__version__}')"
python3 -c "import tqdm; print(f'✅ TQDM {tqdm.__version__}')"

# Check CUDA availability
echo "🎮 Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x run_test_experiments.sh
chmod +x run_experiments_scale.sh
echo "✅ Scripts made executable"

# Check data directories
echo "📁 Checking data directories..."
TEXT_BASE_DIR="/Users/atharvnaphade/Downloads/atharv"
DEEPSEEK_DIR="${TEXT_BASE_DIR}/deepseek-qwen"
QWEN_DIR="${TEXT_BASE_DIR}/qwen-instruct"

if [ -d "${DEEPSEEK_DIR}" ]; then
    DEEPSEEK_COUNT=$(ls -1 "${DEEPSEEK_DIR}"/*.txt 2>/dev/null | wc -l)
    echo "✅ DeepSeek directory found: ${DEEPSEEK_COUNT} text files"
else
    echo "❌ DeepSeek directory not found: ${DEEPSEEK_DIR}"
fi

if [ -d "${QWEN_DIR}" ]; then
    QWEN_COUNT=$(ls -1 "${QWEN_DIR}"/*.txt 2>/dev/null | wc -l)
    echo "✅ Qwen directory found: ${QWEN_COUNT} text files"
else
    echo "❌ Qwen directory not found: ${QWEN_DIR}"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Run test experiments: ./run_test_experiments.sh"
echo "2. If tests pass, run full experiments: ./run_experiments_scale.sh"
echo ""
echo "📚 For more information, see:"
echo "- TEXT_EXPERIMENT_README.md"
echo "- EXPERIMENT_README.md"

