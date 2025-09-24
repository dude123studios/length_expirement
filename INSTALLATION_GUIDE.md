# Installation Guide - Subthought Length Temperature Experiment

This guide will help you set up the environment to run the subthought length temperature experiment with **ZERO ERRORS**.

## 🎯 **Quick Setup (Recommended)**

### **1. Run the Setup Script**
```bash
# Navigate to the experiment directory
cd /Users/atharvnaphade/Code/Research/tester/expirement1/llm-reasoning-activations

# Run the automated setup
./setup_environment.sh
```

This script will:
- ✅ Check Python version
- ✅ Install all required packages
- ✅ Verify package installations
- ✅ Check CUDA availability
- ✅ Make scripts executable
- ✅ Validate data directories

### **2. Run Tests**
```bash
# Test with small examples first
./run_test_experiments.sh
```

### **3. Run Full Experiments**
```bash
# Run at scale
./run_experiments_scale.sh
```

## 🔧 **Manual Setup (Alternative)**

If you prefer manual setup or encounter issues:

### **1. Python Environment**
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Or use conda
conda create -n subthought-exp python=3.11
conda activate subthought-exp
```

### **2. Install Dependencies**
```bash
# Install from requirements file
pip3 install -r requirements.txt

# Or install individually
pip3 install torch>=2.7.1
pip3 install transformers>=4.53.2
pip3 install accelerate>=1.9.0
pip3 install numpy>=2.3.1
pip3 install pandas>=2.0.0
pip3 install scipy>=1.11.0
pip3 install scikit-learn>=1.7.1
pip3 install plotly>=6.3.0
pip3 install matplotlib>=3.10.3
pip3 install tqdm>=4.66.0
pip3 install pyyaml>=6.0.2
pip3 install datasets>=4.0.0
pip3 install gradio>=5.40.0
```

### **3. Verify Installation**
```bash
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import transformers; print('Transformers:', transformers.__version__)"
python3 -c "import plotly; print('Plotly:', plotly.__version__)"
python3 -c "import pandas; print('Pandas:', pandas.__version__)"
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
python3 -c "import scipy; print('SciPy:', scipy.__version__)"
```

## 📋 **System Requirements**

### **Minimum Requirements:**
- **Python**: 3.11+ (as specified in `.python-version`)
- **RAM**: 16GB+ (for 7B models)
- **Storage**: 10GB+ free space
- **GPU**: Recommended (CUDA-compatible)

### **Recommended:**
- **Python**: 3.11 or 3.12
- **RAM**: 32GB+
- **GPU**: NVIDIA GPU with 16GB+ VRAM
- **Storage**: 50GB+ free space

## 🚨 **Common Issues & Solutions**

### **Issue 1: "No module named 'torch'"**
```bash
# Solution: Install PyTorch
pip3 install torch torchvision torchaudio
```

### **Issue 2: "CUDA out of memory"**
```bash
# Solution: Reduce batch size or use CPU
# Edit the experiment script to use smaller max_new_tokens
```

### **Issue 3: "Permission denied" on scripts**
```bash
# Solution: Make scripts executable
chmod +x *.sh
```

### **Issue 4: "No such file or directory" for data**
```bash
# Solution: Check data paths in scripts
# Update paths in run_test_experiments.sh and run_experiments_scale.sh
```

### **Issue 5: "Model not found"**
```bash
# Solution: Ensure you have access to the models
# Models used: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, Qwen/Qwen2.5-7B-Instruct
```

## 🔍 **Verification Checklist**

Before running experiments, verify:

- [ ] Python 3.11+ installed
- [ ] All packages installed (`pip3 list`)
- [ ] Scripts are executable (`ls -la *.sh`)
- [ ] Data directories exist (`ls -la /Users/atharvnaphade/Downloads/atharv/`)
- [ ] Models are accessible (test with small example)
- [ ] GPU available (optional but recommended)

## 📊 **Package Versions**

The experiment has been tested with these versions:

```
torch==2.7.1
transformers==4.53.2
accelerate==1.9.0
numpy==2.3.1
pandas==2.0.0
scipy==1.11.0
scikit-learn==1.7.1
plotly==6.3.0
matplotlib==3.10.3
tqdm==4.66.0
pyyaml==6.0.2
datasets==4.0.0
gradio==5.40.0
```

## 🎯 **Testing Your Setup**

### **Quick Test:**
```bash
# Test imports
python3 -c "
import torch
import transformers
import plotly
import pandas
import numpy
import scipy
print('✅ All imports successful!')
"
```

### **Model Test:**
```bash
# Test model loading (small test)
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print('✅ Model loading successful!')
"
```

## 🚀 **Ready to Run!**

Once setup is complete:

1. **Test**: `./run_test_experiments.sh`
2. **Scale**: `./run_experiments_scale.sh`
3. **Results**: Check `experiment_results_scale/` directory

## 📞 **Getting Help**

If you encounter issues:

1. **Check logs**: Look at terminal output for error messages
2. **Verify setup**: Run `./setup_environment.sh` again
3. **Test imports**: Use the verification commands above
4. **Check data**: Ensure your text files are in the right directories

## 🎉 **Success Indicators**

You'll know setup is successful when:
- ✅ All packages import without errors
- ✅ Scripts run without permission errors
- ✅ Test experiments complete successfully
- ✅ HTML visualizations are generated
- ✅ Statistical results are produced

Your experiment is ready to run at scale! 🚀

