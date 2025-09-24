# 🚀 Complete Subthought Length Experiment Runner

**One script to rule them all!** This script handles everything from environment setup to running the complete experiment.

## 🎯 **Super Simple Usage:**

### **Test Experiment (2 examples, 5-10 minutes):**
```bash
./run_complete_experiment.sh test
```

### **Full-Scale Experiment (100 examples, 45-90 minutes):**
```bash
./run_complete_experiment.sh scale
```

## ✨ **What This Script Does Automatically:**

### **🔧 Environment Setup:**
- ✅ Checks Python version
- ✅ Creates virtual environment (if needed)
- ✅ Activates virtual environment
- ✅ Upgrades pip to latest version

### **📦 Package Installation:**
- ✅ Installs all required packages from `requirements.txt`
- ✅ Verifies all packages are working correctly
- ✅ Checks CUDA availability for GPU acceleration

### **📁 Data Validation:**
- ✅ Checks if trace files exist in the correct location
- ✅ Counts available files
- ✅ Warns if insufficient files for the experiment type

### **🧪 Experiment Execution:**
- ✅ Runs the experiment with optimal parameters
- ✅ Shows real-time progress
- ✅ Handles errors gracefully

### **📊 Results Processing:**
- ✅ Generates publication-ready plots
- ✅ Creates statistical summaries
- ✅ Provides detailed results analysis

## 🎨 **Features:**

### **🌈 Colored Output:**
- 🔵 **Blue:** Information and status updates
- 🟢 **Green:** Success messages
- 🟡 **Yellow:** Warnings
- 🔴 **Red:** Errors
- 🟣 **Purple:** Section headers

### **📈 Progress Tracking:**
- Real-time status updates
- Clear section divisions
- Detailed error messages
- Success confirmations

### **🔍 Comprehensive Checks:**
- Python version validation
- Package installation verification
- Data directory validation
- CUDA availability detection

## 📋 **Prerequisites:**

### **System Requirements:**
- **Python 3.8+** installed
- **Internet connection** for package downloads
- **~5GB disk space** for packages and models
- **Trace files** in `/Users/atharvnaphade/Downloads/atharv/deepseek-qwen/`

### **Optional (for better performance):**
- **CUDA-compatible GPU** for faster processing
- **jq** for enhanced result parsing

## 🚀 **Quick Start:**

```bash
# Navigate to experiment directory
cd /Users/atharvnaphade/Code/Research/tester/expirement1/llm-reasoning-activations

# Run test experiment
./run_complete_experiment.sh test

# If test works, run full experiment
./run_complete_experiment.sh scale
```

## 📊 **Expected Output:**

### **Test Experiment:**
```
🚀 Complete Subthought Length Temperature Experiment
================================
[INFO] Experiment type: test
[SUCCESS] Python version: 3.9.6
[SUCCESS] Virtual environment created
[SUCCESS] Requirements installed successfully
[SUCCESS] PyTorch 2.1.0
[SUCCESS] Transformers 4.35.0
[SUCCESS] DeepSeek directory found: 150 text files
[SUCCESS] Experiment completed successfully!
[SUCCESS] Results:
  Regular temp (T=0.7): 21.0 tokens
  High temp (T=2.5): 13.0 tokens
  Difference: -8.0 tokens
  P-value: 0.079
  Significant: false
```

### **Scale Experiment:**
```
🚀 Complete Subthought Length Temperature Experiment
================================
[INFO] Experiment type: scale
[SUCCESS] Experiment completed successfully!
[SUCCESS] Results:
  Regular temp (T=0.7): 45.2 tokens
  High temp (T=2.5): 32.1 tokens
  Difference: -13.1 tokens
  P-value: 0.023
  Significant: true
```

## 📁 **Generated Files:**

### **Test Results:**
- `test_final_results/subthought_length_results.json` - Raw data
- `test_final_results/main_comparison_publication.pdf` - Main figure
- `test_final_results/probability_evolution_publication.pdf` - Mechanism
- `test_final_results/statistical_summary_publication.pdf` - Statistics

### **Scale Results:**
- `final_results_YYYYMMDD_HHMMSS/subthought_length_results.json` - Raw data
- `final_results_YYYYMMDD_HHMMSS/main_comparison_publication.pdf` - Main figure
- `final_results_YYYYMMDD_HHMMSS/probability_evolution_publication.pdf` - Mechanism
- `final_results_YYYYMMDD_HHMMSS/statistical_summary_publication.pdf` - Statistics

## ⏱️ **Time Estimates:**

| Experiment | Examples | Time | Purpose |
|------------|----------|------|---------|
| **Test** | 2 | 5-10 min | Validation |
| **Scale** | 100 | 45-90 min | Publication |

## 🔧 **Troubleshooting:**

### **Common Issues:**

#### **1. Python Not Found:**
```bash
# Install Python 3.8+
brew install python3  # macOS
sudo apt install python3 python3-venv python3-pip  # Ubuntu
```

#### **2. Permission Denied:**
```bash
# Make script executable
chmod +x run_complete_experiment.sh
```

#### **3. No Trace Files:**
```bash
# Check if files exist
ls /Users/atharvnaphade/Downloads/atharv/deepseek-qwen/*.txt | wc -l
```

#### **4. Package Installation Fails:**
```bash
# Try with no cache
pip install -r requirements.txt --no-cache-dir
```

### **Error Messages:**
- **Red text:** Critical errors that stop execution
- **Yellow text:** Warnings that don't stop execution
- **Blue text:** Information and progress updates
- **Green text:** Success confirmations

## 🎯 **Best Practices:**

### **1. Always Test First:**
```bash
./run_complete_experiment.sh test
```

### **2. Check Results:**
```bash
# Open the PDF files to verify quality
open test_final_results/*.pdf
```

### **3. Run Scale Only After Test:**
```bash
# Only run scale if test works
./run_complete_experiment.sh scale
```

### **4. Monitor Progress:**
- Watch for colored status messages
- Check for error messages in red
- Verify file counts in warnings

## 🎉 **Success Indicators:**

### **✅ Test Success:**
- No red error messages
- "Experiment completed successfully!" in green
- PDF files generated in `test_final_results/`
- Results show reasonable token counts

### **✅ Scale Success:**
- No red error messages
- "Experiment completed successfully!" in green
- PDF files generated in timestamped directory
- Statistical significance achieved (if hypothesis is correct)

## 📚 **Additional Resources:**

- **Detailed README:** `PUBLICATION_PLOTS_README.md`
- **Installation Guide:** `INSTALLATION_GUIDE.md`
- **Experiment Details:** `EXPERIMENT_README.md`

---

**🎯 One script, complete experiment, publication-ready results!**

**Usage:** `./run_complete_experiment.sh [test|scale]`
