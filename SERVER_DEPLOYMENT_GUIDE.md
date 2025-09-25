# GPU Server Deployment Guide for Scale Experiment

This guide provides complete instructions for running the subthought length experiment at scale on a GPU server with comprehensive analysis and beautiful visualizations.

## 🚀 Quick Start

### 1. Upload Files to Server
```bash
# Upload the experiment directory to your GPU server
scp -r /Users/atharvnaphade/Code/Research/tester/expirement1/llm-reasoning-activations/ user@server:/path/to/experiment/
```

### 2. Setup Environment on Server
```bash
# SSH into your server
ssh user@server

# Navigate to experiment directory
cd /path/to/experiment/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Scale Experiment
```bash
# Make scripts executable
chmod +x run_scale_experiment_server.sh
chmod +x comprehensive_analysis.py

# Run the full scale experiment
./run_scale_experiment_server.sh
```

### 4. Run Comprehensive Analysis
```bash
# After experiment completes, run comprehensive analysis
python3 comprehensive_analysis.py --results_file scale_results_*/subthought_length_results.json
```

## 📊 What You'll Get

### **Primary Results:**
- `subthought_length_results.json` - Raw data with statistics
- `main_comparison_publication.pdf` - Main results figure
- `statistical_summary_publication.pdf` - Statistical validation

### **Length Analysis:**
- `length_distribution_histogram.html` - Regular vs High temperature distributions
- `length_difference_histogram.html` - Differences (Regular - High)
- `length_analysis_combined.html` - Combined analysis

### **Probability Analysis:**
- `decision_token_probability_histogram.html` - Probability distributions
- `decision_token_probability_evolution.html` - Probability over time

### **Comprehensive Analysis:**
- `comprehensive_analysis.html` - 6-panel comprehensive analysis
- `iqr_analysis.html` - Detailed IQR analysis
- `comprehensive_statistics.json` - All statistics in JSON format
- `publication_summary.md` - Publication-ready summary

### **Individual Examples:**
- `example_*_probabilities_publication.pdf` - Detailed case studies

## 🔧 Server Requirements

### **Minimum Requirements:**
- **GPU**: 24GB+ VRAM (RTX 4090, A100, H100)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space
- **Python**: 3.8+
- **CUDA**: 11.8+ (for GPU acceleration)

### **Recommended Configuration:**
- **GPU**: 40GB+ VRAM (A100 40GB, H100)
- **RAM**: 64GB+ system memory
- **Storage**: 100GB+ SSD
- **CPU**: 16+ cores

## 📈 Expected Results

### **Sample Size:** 100 examples
### **Statistical Power:** High (sufficient for publication)
### **Time:** 60-120 minutes (depending on GPU)

### **Key Metrics to Check:**
1. **Mean Difference (Regular - High):**
   - Positive: High temperature → shorter reasoning
   - Negative: High temperature → longer reasoning

2. **Statistical Significance:**
   - p < 0.05: Statistically significant
   - p < 0.01: Highly significant

3. **Effect Size (Cohen's d):**
   - 0.2-0.5: Small effect
   - 0.5-0.8: Medium effect
   - >0.8: Large effect

4. **IQR Analysis:**
   - Shows distribution spread
   - Identifies outliers
   - Compares variability

## 🎯 Analysis Workflow

### **Step 1: Run Scale Experiment**
```bash
./run_scale_experiment_server.sh
```

### **Step 2: Run Comprehensive Analysis**
```bash
python3 comprehensive_analysis.py --results_file scale_results_*/subthought_length_results.json
```

### **Step 3: Review Results**
1. Check `publication_summary.md` for key findings
2. Open `comprehensive_analysis.html` for visual analysis
3. Review `iqr_analysis.html` for distribution details
4. Examine individual example plots

### **Step 4: Download Results**
```bash
# Download results to local machine
scp -r user@server:/path/to/experiment/scale_results_* ./
```

## 📊 Publication-Ready Outputs

### **Main Figure:**
- `main_comparison_publication.pdf` - Primary results (300 DPI)
- Shows box plots, scatter plots, statistical annotations

### **Supplementary Figures:**
- `statistical_summary_publication.pdf` - Statistical validation
- `probability_evolution_publication.pdf` - Mechanism analysis
- `comprehensive_analysis.html` - Interactive exploration

### **Data Files:**
- `subthought_length_results.json` - Complete raw data
- `comprehensive_statistics.json` - All calculated statistics

### **Summary Document:**
- `publication_summary.md` - Ready-to-use results summary

## 🔍 Troubleshooting

### **Memory Issues:**
```bash
# Check GPU memory
nvidia-smi

# Monitor during experiment
watch -n 1 nvidia-smi
```

### **Model Loading Issues:**
```bash
# Test model loading
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')"
```

### **Permission Issues:**
```bash
# Fix permissions
chmod +x *.sh
chmod +x *.py
```

## 📋 Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Setup | 10-15 min | Environment setup, dependency installation |
| Model Loading | 5-10 min | Download and load DeepSeek model |
| Experiment | 60-120 min | Run 100 examples with 128 max tokens |
| Analysis | 5-10 min | Generate comprehensive analysis |
| **Total** | **80-155 min** | **Complete workflow** |

## 🎉 Success Indicators

### **Experiment Success:**
- ✅ No error messages during execution
- ✅ 100 examples processed successfully
- ✅ All visualization files generated
- ✅ Statistical significance achieved

### **Analysis Success:**
- ✅ Comprehensive statistics calculated
- ✅ IQR analysis completed
- ✅ Publication summary generated
- ✅ All plots created successfully

## 📞 Support

If you encounter issues:
1. Check the server logs for error messages
2. Verify GPU memory availability
3. Ensure all dependencies are installed
4. Check file permissions

The experiment is designed to be robust and provide comprehensive analysis suitable for publication! 🚀
