# Subthought Length Temperature Experiment (Text Version)

This version of the experiment works with directories containing only text output files (like your `atharv` directory), making it perfect for analyzing existing reasoning traces without needing the full activations format.

## 🎯 **What This Solves**

Your `atharv` directory contains:
- `/Users/atharvnaphade/Downloads/atharv/deepseek-qwen/` - 500 text files
- `/Users/atharvnaphade/Downloads/atharv/qwen-instruct/` - 500 text files

But the original experiment expects:
- `metadata.jsonl`
- `example_{idx}_activations.pt`
- `example_{idx}_output_ids.pt`

**This text version works directly with your existing text files!**

## 📁 **Files Created**

### **Core Experiment Files:**
- `experiment_subthought_length_text.py` - Main experiment script (text version)
- `convert_text_to_activations.py` - Converter for full activations format (optional)

### **Scale-Ready Scripts:**
- `run_test_experiments.sh` - Small-scale tests (5 examples each)
- `run_experiments_scale.sh` - Full-scale experiments (500 examples each)

### **Documentation:**
- `TEXT_EXPERIMENT_README.md` - This guide

## 🚀 **Quick Start**

### **1. Test First (Recommended)**
```bash
# Run small tests to validate everything works
./run_test_experiments.sh
```

This will:
- Test with 5 examples from each model
- Generate small result files
- Validate the setup before running at scale

### **2. Run Full Experiments**
```bash
# Run full-scale experiments
./run_experiments_scale.sh
```

This will:
- Process 500 examples from each model
- Generate comprehensive results
- Create detailed visualizations

## 📊 **What the Experiment Does**

### **For Each Model (DeepSeek & Qwen):**

1. **Loads your text files** from the `atharv` directories
2. **Identifies decision tokens** in the reasoning traces (So, Let, Hmm, I, Okay, First, Wait, But, Now, Then, Since, Therefore, If, Maybe, To)
3. **Continues generation** from just before each decision token:
   - **Greedy sampling** (temperature=0.0)
   - **High-temperature sampling** (temperature=1.2)
4. **Measures subthought length** until the next decision token
5. **Compares results** statistically

### **Expected Hypothesis:**
- High-temperature sampling → longer, more exploratory subthoughts
- Greedy sampling → shorter, more direct paths to decisions

## 📈 **Output Files**

Each experiment generates:

### **Data Files:**
- `subthought_length_results_math.json` - Raw experiment results
- `analysis_statistics.json` - Statistical analysis

### **Visualizations:**
- `subthought_length_boxplot.html` - Distribution comparison
- `subthought_length_scatter.html` - Greedy vs High-temp scatter plot
- `length_difference_histogram.html` - Difference distribution
- `complete_analysis.html` - Combined dashboard

## 🔧 **Manual Usage**

If you want to run experiments manually:

```bash
# Test with DeepSeek
python experiment_subthought_length_text.py \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --benchmark_name "math" \
    --text_dir "/Users/atharvnaphade/Downloads/atharv/deepseek-qwen" \
    --num_examples 10 \
    --high_temperature 1.2 \
    --output_dir "test_deepseek"

# Test with Qwen
python experiment_subthought_length_text.py \
    --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
    --benchmark_name "math" \
    --text_dir "/Users/atharvnaphade/Downloads/atharv/qwen-instruct" \
    --num_examples 10 \
    --high_temperature 1.2 \
    --output_dir "test_qwen"
```

## 📋 **Parameters**

- `--model_name_or_path`: Model to use (DeepSeek or Qwen)
- `--text_dir`: Directory with your text files
- `--num_examples`: Number of examples to process (5 for tests, 500 for full scale)
- `--high_temperature`: Temperature for high-temp sampling (default: 1.2)
- `--max_new_tokens`: Max tokens to generate (default: 128)
- `--output_dir`: Where to save results

## 🎯 **Expected Results**

### **If Hypothesis is Correct:**
```
Analysis Results:
Number of comparisons: 150
Greedy mean length: 12.3 ± 8.1
High-temp mean length: 18.7 ± 12.4
Mean difference: 6.4 ± 9.2
Paired t-test p-value: 0.000001
Effect size (Cohen's d): 0.623
```

### **Interpretation:**
- **Positive difference**: High-temp produces longer subthoughts
- **Low p-value**: Statistically significant difference
- **Large effect size**: Practically significant difference

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **"No decision tokens found"**:
   - Check that your text files contain reasoning with decision words
   - The experiment looks for: So, Let, Hmm, I, Okay, First, Wait, But, Now, Then, Since, Therefore, If, Maybe, To

2. **Model loading errors**:
   - Ensure you have access to the models
   - Check GPU memory for large models

3. **Empty results**:
   - Verify text directories contain the expected files
   - Check that examples have decision tokens

### **Performance Tips:**
- Start with `--num_examples 5` for testing
- Use smaller `--max_new_tokens` for faster execution
- Monitor GPU memory usage

## 🆚 **Model Comparison**

The scripts will run experiments on both models:

### **DeepSeek-R1-Distill-Qwen-7B:**
- Reasoning-focused model
- May show different temperature sensitivity

### **Qwen2.5-7B-Instruct:**
- General instruction-following model
- Baseline for comparison

## 📊 **Analysis Workflow**

1. **Run tests** → Validate setup
2. **Run full experiments** → Generate data
3. **Compare results** → Open HTML visualizations
4. **Statistical analysis** → Check p-values and effect sizes
5. **Draw conclusions** → Temperature impact on reasoning

## 🎉 **Success Indicators**

- ✅ Tests complete without errors
- ✅ Full experiments process all examples
- ✅ HTML visualizations load correctly
- ✅ Statistical tests show significant differences
- ✅ Results align with temperature hypothesis

This setup allows you to run the subthought length experiment at scale using your existing text data, without needing to regenerate activations!

