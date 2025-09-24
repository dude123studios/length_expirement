#!/usr/bin/env python3
"""
Final Correct Subthought Length Experiment

This experiment:
1. Loads DeepSeek-Qwen reasoning traces
2. Identifies random points that end with decision tokens
3. Continues generation with DeepSeek-R1-Distill-Qwen-7B
4. Compares low vs high temperature subthought lengths
5. Optimized for MacBook MPS and scale
"""

import argparse
import os
import random
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from analysis.token_analysis import SPECIAL_TOKENS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Final Correct Subthought Length Experiment")
    parser.add_argument('--model_name_or_path', default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                       help="Model to use for generation")
    parser.add_argument('--traces_dir', default="/Users/atharvnaphade/Downloads/atharv/deepseek-qwen",
                       help="Directory containing .txt trace files")
    parser.add_argument('--output_dir', default="./final_results",
                       help="Directory to save results")
    parser.add_argument('--num_examples', type=int, default=50,
                       help="Number of examples to process")
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help="Maximum tokens to generate per continuation")
    parser.add_argument('--low_temp', type=float, default=0.7,
                       help="Low temperature for baseline (normal generation)")
    parser.add_argument('--high_temp', type=float, default=1.8,
                       help="High temperature for comparison (high randomness)")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument('--device', type=str, default="auto",
                       help="Device to use (auto, mps, cuda, cpu)")
    return parser.parse_args()

def setup_device(device_arg):
    """Setup device with proper GPU optimization"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
            print("🚀 Using CUDA with float16")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float16
            print("🍎 Using Apple Metal (MPS) with float16")
        else:
            device = torch.device("cpu")
            dtype = torch.float32
            print("💻 Using CPU with float32")
    else:
        device = torch.device(device_arg)
        dtype = torch.float16 if device.type != "cpu" else torch.float32
        print(f"🔧 Using {device} with {dtype}")
    
    return device, dtype

def load_model_and_tokenizer(model_name, device, dtype):
    """Load model optimized for GPU/CPU"""
    print(f"📥 Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use device_map for better GPU utilization
    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",  # Automatic GPU placement
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)
    
    model.eval()
    print(f"✅ Model loaded on {device}")
    return model, tokenizer

def get_decision_token_ids(tokenizer, special_tokens):
    """Get token IDs for decision tokens - ONLY properly capitalized versions"""
    decision_token_ids = {}
    
    print("🔍 Getting decision token IDs (CAPITALIZED ONLY):")
    
    for token_str in special_tokens:
        # Only try capitalized versions to maintain proper capitalization
        attempts = [
            token_str,                    # "So"
            f" {token_str}",             # " So" 
            f"\n{token_str}",            # "\nSo"
            f"{token_str},",             # "So,"
            f" {token_str},",            # " So,"
            f"\n{token_str},",           # "\nSo,"
        ]
        
        found_tokens = []
        for attempt in attempts:
            tokens = tokenizer.encode(attempt, add_special_tokens=False)
            if len(tokens) == 1:
                found_tokens.append((attempt, tokens[0]))
                print(f"  ✅ '{attempt}' -> {tokens[0]}")
        
        if found_tokens:
            # Use the first (most common) tokenization
            decision_token_ids[token_str] = found_tokens[0][1]
        else:
            print(f"  ❌ Could not tokenize '{token_str}' as single token")
    
    print(f"🎯 Found {len(decision_token_ids)} CAPITALIZED decision tokens")
    return decision_token_ids

def load_trace_file(file_path, tokenizer, device):
    """Load and tokenize a trace file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    token_ids = tokens[0].to(device)
    
    return text, token_ids

def find_decision_positions(token_ids, decision_token_ids, max_tokens=500):
    """Find all positions where decision tokens occur in the first max_tokens"""
    token_list = token_ids.tolist()
    # Only look at the first max_tokens for scale
    search_tokens = token_list[:max_tokens]
    positions = []
    decision_token_values = list(decision_token_ids.values())
    
    for pos, token_id in enumerate(search_tokens):
        if token_id in decision_token_values:
            positions.append(pos)
    
    print(f"🎯 Found {len(positions)} decision token positions in first {max_tokens} tokens")
    return positions

def select_random_start_position(decision_positions, min_context=10):
    """Select a random position that ends with a decision token"""
    if not decision_positions:
        return None
    
    # Filter positions that have enough context before them
    valid_positions = [pos for pos in decision_positions if pos >= min_context]
    
    if not valid_positions:
        return None
    
    # Select random position
    selected_pos = random.choice(valid_positions)
    return selected_pos

def generate_continuation(model, tokenizer, prefix_ids, temperature, max_new_tokens, decision_token_ids, device, stop_on_high_likelihood=False):
    """Generate continuation with temperature sampling, tracking decision token probabilities over time"""
    input_ids = prefix_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Get decision token values for quick lookup
    decision_token_values = set(decision_token_ids.values())
    
    # Generate token by token
    generated_tokens = []
    current_ids = input_ids.clone()
    current_attention = attention_mask.clone()
    
    # Track decision token probabilities over time
    decision_token_probs_over_time = []
    decision_token_position = None
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Get model outputs
            outputs = model(
                current_ids,
                attention_mask=current_attention,
                return_dict=True
            )
            
            # Get logits for next token
            next_token_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Track decision token probabilities at this step
            step_probs = {}
            max_decision_prob = 0.0
            max_decision_token = None
            
            for token_str, token_id in decision_token_ids.items():
                if token_id < probs.size(0):  # Make sure token_id is valid
                    prob = probs[token_id].item()
                    step_probs[token_str] = prob
                    if prob > max_decision_prob:
                        max_decision_prob = prob
                        max_decision_token = token_str
            
            decision_token_probs_over_time.append({
                'step': step,
                'max_decision_prob': max_decision_prob,
                'max_decision_token': max_decision_token,
                'all_probs': step_probs
            })
            
            # Check if we should stop due to high likelihood decision token (from step 0, no skipping)
            if stop_on_high_likelihood and max_decision_prob > 0.3:  # High likelihood threshold
                decision_token_position = step + 1  # 1-indexed position
                print(f"🎯 Stopped at high-likelihood decision token '{max_decision_token}' at position {decision_token_position} (prob: {max_decision_prob:.3f})")
                break  # Exit the loop, don't generate more tokens
            
            # Generate next token with temperature sampling
            if temperature > 0.0:
                # Sample with temperature
                temp_probs = torch.softmax(next_token_logits / temperature, dim=-1)
                new_token_id = torch.multinomial(temp_probs, num_samples=1).item()
                sampling_method = f"temp_{temperature}"
            else:
                # Greedy sampling
                new_token_id = torch.argmax(next_token_logits).item()
                sampling_method = "greedy"
            
            generated_tokens.append(new_token_id)
            
            # Look for decision token in generated tokens (only if not using high likelihood stopping)
            if not stop_on_high_likelihood and new_token_id in decision_token_values:
                decision_token_position = step + 1  # 1-indexed position
                # Find which decision token this is
                for token_str, token_val in decision_token_ids.items():
                    if token_val == new_token_id:
                        print(f"🎯 Found decision token '{token_str}' at position {decision_token_position} (method: {sampling_method})")
                        break
                break
            
            # Update for next iteration
            current_ids = torch.cat([current_ids, torch.tensor([[new_token_id]], device=device)], dim=1)
            current_attention = torch.ones_like(current_ids)
        else:
            print(f"⚠️ No decision token found in {max_new_tokens} tokens")
    
    # Return the position and probability tracking data
    return decision_token_position if decision_token_position else max_new_tokens, decision_token_probs_over_time

def run_single_experiment(model, tokenizer, trace_file, decision_token_ids, args, device):
    """Run experiment on a single trace file"""
    try:
        # Load trace
        text, token_ids = load_trace_file(trace_file, tokenizer, device)
        
        # Find decision positions in first 500 tokens for scale
        decision_positions = find_decision_positions(token_ids, decision_token_ids, max_tokens=500)
        
        if not decision_positions:
            return None
        
        # Select random start position
        start_pos = select_random_start_position(decision_positions)
        if start_pos is None:
            return None
        
        # Get prefix (everything up to and including the decision token)
        prefix_ids = token_ids[:start_pos + 1]
        
        print(f"📋 Starting from position {start_pos} (prefix length: {len(prefix_ids)})")
        
        # First generation: Regular temperature (0.7) with high likelihood stopping
        regular_position, regular_probs = generate_continuation(
            model, tokenizer, prefix_ids, temperature=args.low_temp, 
            max_new_tokens=args.max_new_tokens, decision_token_ids=decision_token_ids, 
            device=device, stop_on_high_likelihood=True
        )
        
        # Second generation: High temperature (1.8) with high likelihood stopping
        high_temp_position, high_temp_probs = generate_continuation(
            model, tokenizer, prefix_ids, temperature=args.high_temp, 
            max_new_tokens=args.max_new_tokens, decision_token_ids=decision_token_ids, 
            device=device, stop_on_high_likelihood=True
        )
        
        return {
            'file': os.path.basename(trace_file),
            'start_position': int(start_pos),
            'regular_position': regular_position,
            'high_temp_position': high_temp_position,
            'regular_probs_over_time': regular_probs,
            'high_temp_probs_over_time': high_temp_probs,
            'prefix_text': tokenizer.decode(prefix_ids, skip_special_tokens=True)[-100:]  # Last 100 chars
        }
        
    except Exception as e:
        print(f"❌ Error processing {trace_file}: {e}")
        return None

def validate_experiment_setup(model, tokenizer, decision_token_ids, device):
    """Validate that the experiment setup is correct"""
    print(f"\n🔍 EXPERIMENT VALIDATION:")
    
    # Test decision token detection
    test_text = "So let me think about this problem."
    test_tokens = tokenizer.encode(test_text, add_special_tokens=False, return_tensors="pt").to(device)
    
    # Check if we can find decision tokens
    found_tokens = []
    for i, token_id in enumerate(test_tokens[0]):
        if token_id.item() in decision_token_ids.values():
            token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True)
            found_tokens.append((i, token_text, token_id.item()))
    
    print(f"   Decision token detection test: {len(found_tokens)} tokens found")
    for pos, text, token_id in found_tokens:
        print(f"     Position {pos}: '{text}' (ID: {token_id})")
    
    # Test model generation
    try:
        with torch.no_grad():
            outputs = model(test_tokens, return_dict=True)
        print(f"   Model generation test: ✅ Working")
    except Exception as e:
        print(f"   Model generation test: ❌ Failed - {e}")
        return False
    
    # Test temperature sampling
    try:
        logits = outputs.logits[0, -1, :]
        probs_07 = torch.softmax(logits / 0.7, dim=-1)
        probs_18 = torch.softmax(logits / 1.8, dim=-1)
        print(f"   Temperature sampling test: ✅ Working (0.7: {torch.max(probs_07):.3f}, 1.8: {torch.max(probs_18):.3f})")
    except Exception as e:
        print(f"   Temperature sampling test: ❌ Failed - {e}")
        return False
    
    return True

def run_experiment(args):
    """Main experiment function"""
    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device, dtype = setup_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, device, dtype)
    
    # Get decision token IDs
    decision_token_ids = get_decision_token_ids(tokenizer, SPECIAL_TOKENS)
    
    # Validate experiment setup
    if not validate_experiment_setup(model, tokenizer, decision_token_ids, device):
        print("❌ Experiment validation failed. Exiting.")
        return
    
    # Get trace files
    trace_files = [f for f in os.listdir(args.traces_dir) if f.endswith('.txt')]
    trace_files = [os.path.join(args.traces_dir, f) for f in sorted(trace_files)]
    
    print(f"📁 Found {len(trace_files)} trace files")
    print(f"🎯 Processing {min(args.num_examples, len(trace_files))} examples")
    
    # Run experiments
    results = []
    start_time = time.time()
    
    for i, trace_file in enumerate(trace_files[:args.num_examples]):
        print(f"🔄 Processing {i+1}/{min(args.num_examples, len(trace_files))}: {os.path.basename(trace_file)}")
        
        result = run_single_experiment(model, tokenizer, trace_file, decision_token_ids, args, device)
        if result:
            results.append(result)
            print(f"   ✅ Regular: {result['regular_position']}, High-temp: {result['high_temp_position']}")
        else:
            print(f"   ⚠️ Skipped (no valid decision tokens)")
    
    if not results:
        print("❌ No valid results collected!")
        # Still save empty results for debugging
        results_data = {
            'args': vars(args),
            'summary': {'error': 'No valid results collected'},
            'results': []
        }
        results_file = os.path.join(args.output_dir, 'subthought_length_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"💾 Empty results saved to {results_file}")
        return
    
    # Analysis
    regular_positions = [r['regular_position'] for r in results]
    high_temp_positions = [r['high_temp_position'] for r in results]
    
    regular_avg = np.mean(regular_positions)
    high_temp_avg = np.mean(high_temp_positions)
    regular_std = np.std(regular_positions)
    high_temp_std = np.std(high_temp_positions)
    
    # Statistical test
    t_stat, p_value = stats.ttest_rel(high_temp_positions, regular_positions)
    
    # Data integrity checks
    print(f"\n🔍 DATA INTEGRITY CHECKS:")
    print(f"   Sample size: {len(results)}")
    print(f"   Regular positions range: {min(regular_positions)} - {max(regular_positions)}")
    print(f"   High-temp positions range: {min(high_temp_positions)} - {max(high_temp_positions)}")
    print(f"   Both methods hit max length: {sum(1 for p in regular_positions if p == args.max_new_tokens)} regular, {sum(1 for p in high_temp_positions if p == args.max_new_tokens)} high-temp")
    
    # Check for suspicious patterns
    shorter_high_temp = sum(1 for r, h in zip(regular_positions, high_temp_positions) if h < r)
    shorter_regular = sum(1 for r, h in zip(regular_positions, high_temp_positions) if r < h)
    equal = sum(1 for r, h in zip(regular_positions, high_temp_positions) if r == h)
    
    print(f"   High-temp shorter: {shorter_high_temp}, Regular shorter: {shorter_regular}, Equal: {equal}")
    
    print(f"\n📊 RESULTS:")
    print(f"   Regular temp (T={args.low_temp}): {regular_avg:.2f} ± {regular_std:.2f}")
    print(f"   High temp (T={args.high_temp}): {high_temp_avg:.2f} ± {high_temp_std:.2f}")
    print(f"   Difference: {high_temp_avg - regular_avg:.2f}")
    print(f"   T-test p-value: {p_value:.4f}")
    print(f"   Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Warning if results seem too good to be true
    if p_value < 0.01 and abs(high_temp_avg - regular_avg) > 10:
        print(f"⚠️  WARNING: Very significant result with large effect size. Verify this is not due to:")
        print(f"   - Insufficient sample size")
        print(f"   - Data preprocessing issues")
        print(f"   - Model loading problems")
        print(f"   - Decision token detection errors")
    
    # Save results (convert numpy types to Python types for JSON serialization)
    results_data = {
        'args': vars(args),
        'summary': {
            'num_examples': int(len(results)),
            'regular_avg': float(regular_avg),
            'high_temp_avg': float(high_temp_avg),
            'regular_std': float(regular_std),
            'high_temp_std': float(high_temp_std),
            'difference': float(high_temp_avg - regular_avg),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        },
        'results': results
    }
    
    results_file = os.path.join(args.output_dir, 'subthought_length_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"💾 Results saved to {results_file}")
    
    # Create visualizations
    create_visualizations(results, args.output_dir, regular_avg, high_temp_avg, args.low_temp, args.high_temp)
    
    # Create decision token probability histograms
    create_decision_token_probability_histograms(results, args.output_dir, args.low_temp, args.high_temp)
    
    # Create publication-ready plots
    try:
        from publication_plots import create_publication_plots
        create_publication_plots(results, args.output_dir, args.low_temp, args.high_temp, args.model_name_or_path)
    except ImportError as e:
        print(f"⚠️ Could not import publication_plots: {e}")
        print("📊 Using standard plots instead")
    
    elapsed = time.time() - start_time
    print(f"⏱️ Total time: {elapsed:.1f}s")

def create_visualizations(results, output_dir, regular_avg, high_temp_avg, regular_temp, high_temp):
    """Create visualization plots"""
    regular_positions = [r['regular_position'] for r in results]
    high_temp_positions = [r['high_temp_position'] for r in results]
    
    # Box plot
    fig = go.Figure()
    fig.add_trace(go.Box(y=regular_positions, name=f'Regular Temp (T={regular_temp})', boxpoints='outliers'))
    fig.add_trace(go.Box(y=high_temp_positions, name=f'High Temp (T={high_temp})', boxpoints='outliers'))
    
    fig.update_layout(
        title='Decision Token Positions: Regular vs High Temperature',
        yaxis_title='Position of First Decision Token (after 5 tokens)',
        xaxis_title='Temperature Setting',
        showlegend=True
    )
    
    boxplot_file = os.path.join(output_dir, 'decision_token_position_boxplot.html')
    fig.write_html(boxplot_file)
    print(f"📊 Box plot saved to {boxplot_file}")
    
    # Scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=regular_positions, 
        y=high_temp_positions,
        mode='markers',
        marker=dict(size=8, opacity=0.7),
        text=[r['file'] for r in results],
        hovertemplate='<b>%{text}</b><br>Regular: %{x}<br>High-temp: %{y}<extra></extra>'
    ))
    
    # Add diagonal line
    max_val = max(max(regular_positions), max(high_temp_positions))
    fig.add_trace(go.Scatter(
        x=[0, max_val], 
        y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Equal Positions'
    ))
    
    fig.update_layout(
        title='Decision Token Positions: Regular vs High Temperature (Paired)',
        xaxis_title=f'Regular Temperature (T={regular_temp})',
        yaxis_title=f'High Temperature (T={high_temp})',
        showlegend=True
    )
    
    scatter_file = os.path.join(output_dir, 'decision_token_position_scatter.html')
    fig.write_html(scatter_file)
    print(f"📊 Scatter plot saved to {scatter_file}")
    
    # Histogram of differences
    differences = [h - r for h, r in zip(high_temp_positions, regular_positions)]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=differences, nbinsx=20, name='Position Differences'))
    
    fig.update_layout(
        title='Distribution of Position Differences (High - Regular Temperature)',
        xaxis_title='Position Difference (tokens)',
        yaxis_title='Frequency',
        showlegend=False
    )
    
    hist_file = os.path.join(output_dir, 'position_difference_histogram.html')
    fig.write_html(hist_file)
    print(f"📊 Histogram saved to {hist_file}")

def create_decision_token_probability_histograms(results, output_dir, regular_temp, high_temp):
    """Create histograms showing decision token probabilities over time"""
    print("📊 Creating decision token probability histograms...")
    
    # Collect all probability data
    regular_probs_data = []
    high_temp_probs_data = []
    
    for result in results:
        if 'regular_probs_over_time' in result and 'high_temp_probs_over_time' in result:
            for step_data in result['regular_probs_over_time']:
                regular_probs_data.append({
                    'step': step_data['step'],
                    'max_prob': step_data['max_decision_prob'],
                    'token': step_data['max_decision_token'],
                    'file': result['file']
                })
            
            for step_data in result['high_temp_probs_over_time']:
                high_temp_probs_data.append({
                    'step': step_data['step'],
                    'max_prob': step_data['max_decision_prob'],
                    'token': step_data['max_decision_token'],
                    'file': result['file']
                })
    
    if not regular_probs_data or not high_temp_probs_data:
        print("⚠️ No probability data available for histograms")
        return
    
    # Create histogram of max decision token probabilities
    fig = go.Figure()
    
    # Regular temperature probabilities
    regular_probs = [d['max_prob'] for d in regular_probs_data]
    fig.add_trace(go.Histogram(
        x=regular_probs,
        name=f'Regular Temp (T={regular_temp})',
        opacity=0.7,
        nbinsx=20
    ))
    
    # High temperature probabilities
    high_temp_probs = [d['max_prob'] for d in high_temp_probs_data]
    fig.add_trace(go.Histogram(
        x=high_temp_probs,
        name=f'High Temp (T={high_temp})',
        opacity=0.7,
        nbinsx=20
    ))
    
    fig.update_layout(
        title='Distribution of Maximum Decision Token Probabilities',
        xaxis_title='Maximum Decision Token Probability',
        yaxis_title='Frequency',
        barmode='overlay',
        showlegend=True
    )
    
    prob_hist_file = os.path.join(output_dir, 'decision_token_probability_histogram.html')
    fig.write_html(prob_hist_file)
    print(f"📊 Probability histogram saved to {prob_hist_file}")
    
    # Create line plot showing probability evolution over time
    fig = go.Figure()
    
    # Average probabilities over time for regular temp
    regular_steps = {}
    for d in regular_probs_data:
        step = d['step']
        if step not in regular_steps:
            regular_steps[step] = []
        regular_steps[step].append(d['max_prob'])
    
    regular_avg_probs = []
    regular_steps_sorted = sorted(regular_steps.keys())
    for step in regular_steps_sorted:
        avg_prob = np.mean(regular_steps[step])
        regular_avg_probs.append(avg_prob)
    
    fig.add_trace(go.Scatter(
        x=regular_steps_sorted,
        y=regular_avg_probs,
        mode='lines+markers',
        name=f'Regular Temp (T={regular_temp})',
        line=dict(width=2)
    ))
    
    # Average probabilities over time for high temp
    high_temp_steps = {}
    for d in high_temp_probs_data:
        step = d['step']
        if step not in high_temp_steps:
            high_temp_steps[step] = []
        high_temp_steps[step].append(d['max_prob'])
    
    high_temp_avg_probs = []
    high_temp_steps_sorted = sorted(high_temp_steps.keys())
    for step in high_temp_steps_sorted:
        avg_prob = np.mean(high_temp_steps[step])
        high_temp_avg_probs.append(avg_prob)
    
    fig.add_trace(go.Scatter(
        x=high_temp_steps_sorted,
        y=high_temp_avg_probs,
        mode='lines+markers',
        name=f'High Temp (T={high_temp})',
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title='Decision Token Probability Evolution Over Time',
        xaxis_title='Generation Step',
        yaxis_title='Average Maximum Decision Token Probability',
        showlegend=True
    )
    
    prob_evolution_file = os.path.join(output_dir, 'decision_token_probability_evolution.html')
    fig.write_html(prob_evolution_file)
    print(f"📊 Probability evolution plot saved to {prob_evolution_file}")

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
