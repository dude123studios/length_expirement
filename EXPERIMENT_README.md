# Subthought Length Temperature Experiment

This experiment compares the length of reasoning subthoughts under high-temperature sampling vs. greedy/low-temperature sampling, starting from the same partial reasoning trace up to a decision token.

## Overview

The experiment tests the hypothesis that high-temperature sampling leads to longer, more exploratory reasoning subthoughts before reaching decision tokens, while greedy decoding produces shorter, more direct paths.

## Files

- `experiment_subthought_length.py` - Main experiment script
- `test_experiment.py` - Test script for validation
- `EXPERIMENT_README.md` - This documentation

## Prerequisites

1. **Generated Activations**: You need to have already generated activations using `generate_activations.py`
2. **Model Access**: The model specified in `--model_name_or_path` must be accessible
3. **Dependencies**: All required packages from the existing codebase

## Usage

### Basic Usage

```bash
python experiment_subthought_length.py \
    --model_name_or_path /path/to/your/model \
    --benchmark_name math \
    --activations_dir /path/to/activations \
    --num_examples 100 \
    --high_temperature 1.2 \
    --output_dir experiment_results
```

### Parameters

- `--model_name_or_path` (required): Path to the model (e.g., "Qwen/Qwen2.5-7B-Instruct")
- `--benchmark_name` (default: "math"): Benchmark dataset name
- `--activations_dir` (required): Directory containing stored activations
- `--num_examples` (default: 100): Number of examples to process
- `--high_temperature` (default: 1.2): Temperature for high-temperature sampling
- `--max_new_tokens` (default: 128): Maximum tokens to generate before stopping
- `--output_dir` (default: "experiment_results"): Directory to save results

### Testing

Run a small test first:

```bash
python test_experiment.py \
    --model_name_or_path /path/to/your/model \
    --activations_dir /path/to/activations \
    --benchmark_name math
```

## Experiment Design

### 1. Decision Token Identification
The experiment identifies decision tokens from the `SPECIAL_TOKENS` set:
- "So", "Let", "Hmm", "I", "Okay", "First", "Wait", "But", "Now", "Then"
- "Since", "Therefore", "If", "Maybe", "To"

### 2. Continuation Process
For each decision token position in stored reasoning traces:
1. **Greedy Generation**: Continue with `temperature=0.0` (deterministic)
2. **High-Temperature Generation**: Continue with `temperature=1.2` (or specified value)
3. **Length Measurement**: Count tokens until the next decision token is reached

### 3. Data Collection
Each comparison records:
- Example index and decision position
- Subthought length for both strategies
- Generated text and token sequences
- Length difference (high-temp - greedy)

## Output Files

The experiment generates several output files in the specified directory:

### Data Files
- `subthought_length_results_{benchmark_name}.json` - Raw experiment results
- `analysis_statistics.json` - Statistical analysis results

### Visualizations
- `subthought_length_boxplot.html` - Box plots comparing distributions
- `subthought_length_scatter.html` - Scatter plot of greedy vs high-temp lengths
- `length_difference_histogram.html` - Histogram of length differences
- `complete_analysis.html` - Combined analysis dashboard

## Analysis

### Statistical Tests
- **Paired t-test**: Tests if mean lengths differ significantly
- **Wilcoxon signed-rank test**: Non-parametric alternative
- **Effect size (Cohen's d)**: Measures practical significance

### Key Metrics
- Mean and median subthought lengths for each strategy
- Standard deviations and distributions
- Percentage of cases where high-temp produces longer subthoughts

## Expected Results

### Hypothesis
High-temperature sampling should produce:
- **Longer subthoughts**: More exploratory reasoning before decisions
- **Higher variance**: More diverse reasoning paths
- **Different decision patterns**: Potentially different decision token usage

### Interpretation
- **Positive length difference**: High-temp produces longer subthoughts
- **Negative length difference**: Greedy produces longer subthoughts
- **Large effect size**: Practically significant difference
- **Low p-value**: Statistically significant difference

## Troubleshooting

### Common Issues

1. **No decision tokens found**: 
   - Check that your activations contain the expected special tokens
   - Verify the tokenization matches the SPECIAL_TOKENS set

2. **Model loading errors**:
   - Ensure model path is correct and accessible
   - Check available GPU memory for large models

3. **Empty results**:
   - Verify activations directory contains the expected files
   - Check that examples have decision tokens in their sequences

### Performance Tips

- Start with a small `--num_examples` for testing
- Use smaller `--max_new_tokens` for faster execution
- Monitor GPU memory usage with large models

## Example Results

```
Analysis Results:
Number of comparisons: 150
Greedy mean length: 12.3 ± 8.1
High-temp mean length: 18.7 ± 12.4
Mean difference: 6.4 ± 9.2
Paired t-test p-value: 0.000001
Wilcoxon test p-value: 0.000003
Effect size (Cohen's d): 0.623
```

This would indicate that high-temperature sampling produces significantly longer subthoughts with a medium-to-large effect size.

## Integration with Existing Codebase

The experiment reuses existing components:
- `utils.data_loader` for benchmark loading
- `utils.activations_loader` for stored activations
- `analysis.token_analysis` for special token definitions
- Existing model loading and generation patterns

This ensures consistency with your existing research pipeline while adding the new temperature comparison functionality.

