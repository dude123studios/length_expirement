# 🚀 Super Easy GPU Server Setup

This guide makes it **extremely simple** to run the experiment on a new GPU server.

## 📋 **Method 1: All-in-One (Recommended)**

### **Step 1: Upload Everything**
```bash
# Upload the entire experiment directory to your server
scp -r /Users/atharvnaphade/Code/Research/tester/expirement1/llm-reasoning-activations/ user@your-server:/home/username/experiment/
```

### **Step 2: Upload Data**
```bash
# Upload your data files
scp -r /Users/atharvnaphade/Downloads/atharv/deepseek-qwen/ user@your-server:/home/username/experiment/data/
```

### **Step 3: SSH and Run Everything**
```bash
# SSH into your server and run setup + experiment
ssh user@your-server
cd /home/username/experiment/
./setup_server.sh && ./run_scale_experiment_server.sh
```

## 📋 **Method 2: Using Upload Script**

### **Step 1: Upload Experiment Files**
```bash
scp -r /Users/atharvnaphade/Code/Research/tester/expirement1/llm-reasoning-activations/ user@your-server:/home/username/experiment/
```

### **Step 2: Upload Data (Automated)**
```bash
# Use the upload script (from your local machine)
cd /Users/atharvnaphade/Code/Research/tester/expirement1/llm-reasoning-activations/
./upload_data.sh user@your-server /home/username/experiment/
```

### **Step 3: SSH and Run**
```bash
ssh user@your-server
cd /home/username/experiment/
./setup_server.sh && ./run_scale_experiment_server.sh
```

## 📋 **Method 3: Manual (If Scripts Don't Work)**

### **Step 1: Upload Files**
```bash
scp -r /Users/atharvnaphade/Code/Research/tester/expirement1/llm-reasoning-activations/ user@your-server:/home/username/experiment/
```

### **Step 2: SSH and Setup**
```bash
ssh user@your-server
cd /home/username/experiment/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x run_scale_experiment_server.sh
chmod +x experiment_final_correct.py
```

### **Step 3: Upload Data**
```bash
# From your local machine
scp -r /Users/atharvnaphade/Downloads/atharv/deepseek-qwen/ user@your-server:/home/username/experiment/data/
```

### **Step 4: Run Experiment**
```bash
# Back on the server
./run_scale_experiment_server.sh
```

## 🎯 **What You'll Get**

### **📊 PARAMOUNT: CSV File**
- `scale_experiment_results.csv` with columns:
  - `greedy_length` - Length for greedy (T=0.0)
  - `high_temp_length` - Length for high temperature (T=3.0)  
  - `difference` - Greedy - High temperature

### **📈 Essential Plots**
- `scale_experiment_aggregated_results.html` - 4-panel analysis
- `scale_experiment_summary.html` - Summary statistics

## 🔧 **Server Requirements**
- **GPU:** 24GB+ VRAM (RTX 4090, A100, H100)
- **RAM:** 32GB+ system memory
- **Python:** 3.8+
- **CUDA:** 11.8+ (for GPU acceleration)

## ⏱️ **Timeline**
- **Setup:** 5-10 minutes
- **Experiment:** 60-120 minutes (100 examples)
- **Total:** 65-130 minutes

## 🎉 **Results Location**
```
scale_results_YYYYMMDD_HHMMSS/
├── scale_experiment_results.csv          ← PARAMOUNT
├── scale_experiment_aggregated_results.html
├── scale_experiment_summary.html
└── subthought_length_results.json
```

## 🆘 **Troubleshooting**

### **If Upload Fails:**
```bash
# Check if server is accessible
ping your-server.com

# Check SSH access
ssh user@your-server "echo 'SSH works'"
```

### **If Setup Fails:**
```bash
# Check Python version
python3 --version

# Check GPU
nvidia-smi

# Check memory
free -h
```

### **If Experiment Fails:**
```bash
# Check data directory
ls -la data/deepseek-qwen/

# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

## 🎯 **Quick Commands Summary**

```bash
# Upload everything
scp -r /Users/atharvnaphade/Code/Research/tester/expirement1/llm-reasoning-activations/ user@server:/home/username/experiment/

# Upload data
scp -r /Users/atharvnaphade/Downloads/atharv/deepseek-qwen/ user@server:/home/username/experiment/data/

# SSH and run
ssh user@server
cd /home/username/experiment/
./setup_server.sh && ./run_scale_experiment_server.sh
```

That's it! 🚀
