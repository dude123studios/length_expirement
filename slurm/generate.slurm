#!/bin/bash
#SBATCH --job-name=generate_activations
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=long
#SBATCH --output=/home/mila/p/prateek.humane/scratch/llm-reasoning-activations/logs/%x-%j.out
#SBATCH --error=/home/mila/p/prateek.humane/scratch/llm-reasoning-activations/logs/%x-%j.err

if [[ "$*" == *"--help"* ]]; then
  echo "Usage: sbatch slurm/generate.slurm [options]"
  echo "Options:"
  echo "  --model MODEL            Model name"
  echo "  --data TASK              Benchmark name (e.g. math, aime24)"
  echo "  --config CONIFIG         Configuration suffix (e.g. demo, v00.00)"
  echo "  --args \"ARGS\"          Optional arguments to pass to the training script"
  exit 0
fi

# modules
module load python/3.10 cuda/12.4.1

# venv
# source /home/mila/p/prateek.humane/scratch/llm-reasoning-activations/venv/bin/activate

set -a
# source secret tokens
source /path/to/.env
set +a

export HF_HOME="/home/mila/p/prateek.humane/scratch/hf_hub"
export HF_TOKEN="$HF_TOKEN"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

START_TIME=$(date +%s)
echo "START TIME: $(date)"

# Default values
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TASK="math"
CONFIG="qwen-instruct.yaml"
OUTPUT_DIR="/home/mila/p/prateek.humane/scratch/llm-reasoning-activations/activations"
OPTIONAL_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
  --model)
    MODEL="$2"
    shift 2
    ;;
  --task)
    TASK="$2"
    shift 2
    ;;
  --config)
    CONFIG="$2"
    shift 2
    ;;
  --args)
    OPTIONAL_ARGS="$2"
    shift 2
    ;;
  *)
    echo "Unknown option: $1"
    echo "Use --help for usage information"
    exit 1
    ;;
  esac
done

# Validate required arguments
if [[ -z "$MODEL" || -z "$TASK" || -z "$CONFIG" ]]; then
  echo "Error: Missing required arguments"
  echo "Run with --help for usage information"
  exit 1
fi

# Split OPTIONAL_ARGS into an array for proper expansion
IFS=' ' read -r -a OPTIONAL_ARGS_ARR <<<"$OPTIONAL_ARGS"

echo "Running eval_runner.py with:"
echo "  Model:        $MODEL"
echo "  Task:         $TASK"
echo "  Config file:  $CONFIG_FILE"
echo "  Extra args:   $OPTIONAL_ARGS"

cd ~/C1/llm-reasoning-activations/

# torchrun --nproc-per-node 4 demo.py
uv run generate_activations.py \
  --model_name_or_path "$MODEL" \
  --benchmark_name "$TASK" \
  --sampling_config "$CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  "${OPTIONAL_ARGS_ARR[@]}"

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"

# uv run generate_activations.py \
#   --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
#   --benchmark_name "math" \
#   --sampling_config "qwen-instruct.yaml" \
#   --output_dir "/home/mila/p/prateek.humane/scratch/llm-reasoning-activations/activations"
