#!/bin/bash

# Super simple server setup script
# This handles everything from scratch

set -e  # Exit on any error

echo "🚀 GPU Server Setup Script"
echo "=========================="
echo ""

# Check if we're in the right directory
if [ ! -f "experiment_final_correct.py" ]; then
    echo "❌ Error: experiment_final_correct.py not found"
    echo "Please run this script from the experiment directory"
    exit 1
fi

# Check Python version
echo "📋 Checking Python version..."
python3 --version
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Error: Python 3.8+ required"
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ NVIDIA GPU detected"
else
    echo "⚠️  NVIDIA GPU not detected - will use CPU"
fi

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

# Check key packages
echo "🔍 Verifying key packages..."
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
python3 -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"
python3 -c "import pandas; print(f'✅ Pandas {pandas.__version__}')"

# Check CUDA availability
echo "🎮 Checking CUDA availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.device_count()} GPU(s)')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')
else:
    print('⚠️  CUDA not available - will use CPU')
"

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x run_scale_experiment_server.sh
chmod +x experiment_final_correct.py
echo "✅ Scripts made executable"

# Create data directory
echo "📁 Creating data directory..."
mkdir -p data/deepseek-qwen
echo "✅ Data directory created: data/deepseek-qwen"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📁 Next steps:"
echo "1. Upload your data to: data/deepseek-qwen/"
echo "2. Run: ./run_scale_experiment_server.sh"
echo ""
echo "💡 To upload data from your local machine:"
echo "   scp -r /path/to/your/deepseek-qwen/ user@server:/path/to/experiment/data/"
