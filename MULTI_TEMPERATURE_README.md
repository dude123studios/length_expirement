# 🌡️ Multi-Temperature Subthought Length Experiment

This experiment runs the subthought length analysis across multiple temperatures to understand how temperature affects reasoning length.

## 🎯 **What It Does**

Instead of comparing just two temperatures (greedy vs high), this experiment runs **4 temperatures by default**:
- **T=0.0** (Greedy - deterministic)
- **T=0.7** (Regular - balanced)
- **T=1.5** (High - more random)
- **T=2.5** (Very High - very random)

For each example, it generates continuations at all temperatures and measures how many tokens until the next decision token.

## 📊 **Output**

### **CSV File (PARAMOUNT)**
`multi_temperature_results.csv` with columns:
- `example_id` - Example number
- `file` - Source file name
- `start_position` - Where generation started
- `temp_0.0_length` - Tokens until decision at T=0.0
- `temp_0.7_length` - Tokens until decision at T=0.7
- `temp_1.5_length` - Tokens until decision at T=1.5
- `temp_2.5_length` - Tokens until decision at T=2.5

### **Visualizations**
`multi_temperature_analysis.html` - 4-panel analysis:
1. **Length Distribution** - Histograms for each temperature
2. **Temperature Comparison** - Box plots comparing all temperatures
3. **Length vs Temperature** - Scatter plot showing individual examples
4. **Statistical Summary** - Bar chart with means and error bars

## 🚀 **How to Run**

### **Test (2 examples)**
```bash
./test_multi_temperature.sh
```

### **Scale (100 examples)**
```bash
./run_multi_temperature_scale.sh
```

### **Custom Temperatures**
```bash
python3 experiment_multi_temperature.py \
  --traces_dir "./data/deepseek-qwen" \
  --num_examples 50 \
  --temperatures 0.0 0.5 1.0 1.5 2.0 2.5 \
  --output_dir "custom_results"
```

## ⚙️ **Parameters**

- `--temperatures` - List of temperatures to test (default: 0.0 0.7 1.5 2.5)
- `--num_examples` - Number of examples to process (default: 50)
- `--max_new_tokens` - Maximum tokens to generate (default: 128)
- `--traces_dir` - Directory with .txt trace files
- `--output_dir` - Where to save results
- `--seed` - Random seed for reproducibility

## 📈 **Expected Results**

**Hypothesis**: Higher temperatures should lead to **shorter** reasoning before hitting decision tokens, as the model becomes more random and less focused.

**Analysis**: The CSV allows you to:
- Compare any two temperatures
- Calculate correlations between temperature and length
- Perform statistical tests across all temperatures
- Identify patterns in individual examples

## 🔧 **Server Setup**

Same as the regular experiment:

1. **Upload files**:
   ```bash
   scp -r /path/to/experiment/ user@server:/home/username/experiment/
   ```

2. **Upload data**:
   ```bash
   scp -r /path/to/deepseek-qwen/ user@server:/home/username/experiment/data/
   ```

3. **Setup and run**:
   ```bash
   ssh user@server
   cd /home/username/experiment/
   ./setup_server.sh
   ./run_multi_temperature_scale.sh
   ```

## ⏱️ **Timeline**

- **Test**: 5-10 minutes (2 examples × 4 temperatures = 8 generations)
- **Scale**: 2-4 hours (100 examples × 4 temperatures = 400 generations)

## 🎉 **Results Location**

```
multi_temp_scale_results_YYYYMMDD_HHMMSS/
├── multi_temperature_results.csv          ← PARAMOUNT
├── multi_temperature_analysis.html        ← 4-panel analysis
└── multi_temperature_results.json         ← Raw data
```

## 📊 **Analysis Examples**

### **Compare Two Temperatures**
```python
import pandas as pd
df = pd.read_csv('multi_temperature_results.csv')

# Compare greedy vs high temperature
greedy_vs_high = df['temp_0.0_length'] - df['temp_2.5_length']
print(f"Mean difference (Greedy - High): {greedy_vs_high.mean():.2f}")
```

### **Temperature Correlation**
```python
# Calculate correlation between temperature and length
temps = [0.0, 0.7, 1.5, 2.5]
means = [df[f'temp_{t}_length'].mean() for t in temps]
correlation = np.corrcoef(temps, means)[0, 1]
print(f"Temperature-Length correlation: {correlation:.3f}")
```

## 🆚 **vs Regular Experiment**

| Feature | Regular Experiment | Multi-Temperature |
|---------|-------------------|-------------------|
| Temperatures | 2 (0.0, 3.0) | 4+ (customizable) |
| Analysis | Greedy vs High | All temperatures |
| CSV Columns | 3 | 6+ |
| Plots | 2-panel | 4-panel |
| Runtime | 1-2 hours | 2-4 hours |

The multi-temperature experiment gives you **much richer data** for understanding how temperature affects reasoning patterns! 🚀
