# Publication-Ready Plots for Subthought Length Experiment

This document describes the high-quality, publication-ready plots generated for the subthought length experiment comparing regular vs. high-temperature sampling.

## 📊 Generated Plots

### 1. Main Comparison Plot (`main_comparison_publication.pdf`)
**Purpose:** Primary comparison showing the main experimental results

**Contents:**
- **Panel A:** Box plots comparing subthought lengths between regular (T=0.7) and high (T=1.8) temperature
- **Panel B:** Paired scatter plot showing individual example comparisons
- **Statistical annotations:** Mean differences, t-test results, correlation coefficients
- **Features:** Individual data points, mean lines, error bars, grid lines

**Usage:** Main figure for papers, presentations, and reports

### 2. Probability Evolution Plot (`probability_evolution_publication.pdf`)
**Purpose:** Shows how decision token probabilities evolve during generation

**Contents:**
- Line plots showing maximum decision token probability over generation steps
- Separate lines for regular and high-temperature conditions
- Error bars showing standard deviation across examples
- High-likelihood threshold line (0.3) for reference
- Stopping points marked when threshold is reached

**Usage:** Supplementary figure showing the mechanism behind the results

### 3. Statistical Summary Plot (`statistical_summary_publication.pdf`)
**Purpose:** Comprehensive statistical analysis and validation

**Contents:**
- **Panel A:** Histogram of position differences (High - Regular)
- **Panel B:** Q-Q plot for normality testing
- **Panel C:** Box plot comparison of methods
- **Panel D:** Detailed statistical summary with all key metrics

**Usage:** Supplementary material for statistical validation

### 4. Individual Example Plots (`example_*_probabilities_publication.pdf`)
**Purpose:** Detailed view of probability evolution for specific examples

**Contents:**
- Side-by-side plots for regular vs. high-temperature
- Step-by-step probability evolution
- Stopping points clearly marked
- High-likelihood threshold reference

**Usage:** Detailed analysis of individual cases, debugging, or case studies

## 🎨 Design Features

### Publication Standards
- **High resolution:** 300 DPI for crisp printing
- **Vector formats:** PDF for scalability
- **Professional fonts:** Times New Roman for academic papers
- **Consistent styling:** Unified color palette and formatting
- **Clear annotations:** Statistical significance, effect sizes, sample sizes

### Color Palette
- **Regular temperature:** Blue (#2E86AB)
- **High temperature:** Red/Pink (#A23B72)
- **Neutral elements:** Gray (#6C757D)
- **Accent highlights:** Orange (#F18F01)
- **Text:** Dark gray (#212529)

### Typography
- **Titles:** 16pt, bold
- **Axis labels:** 14pt, bold
- **Legend:** 12pt
- **Annotations:** 10-12pt
- **Statistical text:** Monospace font for clarity

## 📈 Statistical Information Included

### Descriptive Statistics
- Mean ± Standard deviation
- Median and range
- Sample size
- Effect size (Cohen's d)

### Inferential Statistics
- Paired t-test results
- Correlation coefficients
- P-values with appropriate precision
- Confidence intervals (where applicable)

### Data Quality Checks
- Normality testing (Q-Q plots)
- Outlier identification
- Distribution analysis
- Missing data indicators

## 🚀 Usage Instructions

### Generate Plots from Existing Results
```bash
python3 generate_publication_plots.py \
  --results_file test_final_results/subthought_length_results.json \
  --output_dir test_final_results \
  --model_name "DeepSeek-R1-Distill-Qwen-7B"
```

### Generate Plots During Experiment
The experiment automatically generates publication plots when run:
```bash
python3 experiment_final_correct.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --traces_dir "/path/to/traces" \
  --output_dir "./results" \
  --num_examples 100
```

### Customize Plots
Edit `publication_plots.py` to modify:
- Color schemes
- Font sizes and families
- Statistical annotations
- Plot layouts and dimensions

## 📝 Citation and Usage

### For Academic Papers
When using these plots in academic papers, please cite:
- The experimental methodology
- The model used (DeepSeek-R1-Distill-Qwen-7B)
- The statistical methods employed
- The decision token detection approach

### File Formats
- **PDF:** Use for papers, presentations, and print media
- **PNG:** Use for web, social media, and digital presentations
- **HTML:** Use for interactive exploration and web-based reports

## 🔧 Technical Requirements

### Dependencies
```bash
pip install matplotlib seaborn pandas numpy scipy plotly
```

### System Requirements
- Python 3.8+
- Sufficient memory for large datasets
- High-resolution display for preview

## 📊 Plot Specifications

### Dimensions
- **Main comparison:** 12×5 inches
- **Probability evolution:** 10×6 inches
- **Statistical summary:** 12×10 inches
- **Individual examples:** 12×5 inches

### Resolution
- **Print quality:** 300 DPI
- **Web quality:** 150 DPI
- **Vector format:** Scalable to any size

### File Sizes
- **PDF files:** ~500KB - 2MB
- **PNG files:** ~1MB - 5MB
- **HTML files:** ~100KB - 500KB

## 🎯 Best Practices

### For Papers
1. Use PDF format for submission
2. Include statistical annotations
3. Ensure sufficient contrast for grayscale printing
4. Provide figure captions with detailed descriptions

### For Presentations
1. Use PNG format for slides
2. Increase font sizes for readability
3. Use high contrast colors
4. Include key statistics in slide text

### For Reports
1. Use HTML format for interactive exploration
2. Include hover information
3. Provide downloadable data
4. Add explanatory text

## 🐛 Troubleshooting

### Common Issues
1. **Missing dependencies:** Install seaborn and matplotlib
2. **Font issues:** Ensure Times New Roman is available
3. **Memory errors:** Reduce dataset size or increase system memory
4. **Permission errors:** Check write permissions for output directory

### Support
For issues with plot generation, check:
1. Python version compatibility
2. Package versions in requirements.txt
3. File permissions and paths
4. Data format and structure

---

**Generated by:** Subthought Length Experiment Analysis Pipeline  
**Version:** 1.0  
**Last Updated:** 2024
