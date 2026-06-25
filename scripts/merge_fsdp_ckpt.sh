#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=sid-merge-fsdp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output/%x-%j.out

# Usage:
#   sbatch --export=CKPT_DIR=/path/to/actor,MERGED_DIR=/durable/model scripts/merge_fsdp_ckpt.sh

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_DIR"

source ./scripts/snellius_env.sh

CKPT_DIR="${1:-${CKPT_DIR:-}}"
MERGED_DIR="${2:-${MERGED_DIR:-}}"

if [[ -z "$CKPT_DIR" || -z "$MERGED_DIR" ]]; then
    echo "ERROR: CKPT_DIR and MERGED_DIR are required."
    echo "Usage: bash scripts/merge_fsdp_ckpt.sh /path/to/actor /durable/output"
    exit 1
fi

CKPT_DIR="${CKPT_DIR%/}"

echo "Verl checkpoint directory: $CKPT_DIR"
echo "Will save merged HF model to: $MERGED_DIR"
${PYTHON_CMD} ./scripts/merge_fsdp_checkpoint.py \
    --checkpoint "$CKPT_DIR" \
    --output-dir "$MERGED_DIR" \
    --base-model "${BASE_MODEL:-/home/scur1249/Office_Products_checkpoint/merged}" \
    --mode "${MERGE_MODE:-auto}" \
    --use-cpu-init

echo "Merge completed."
echo "Merged Hugging Face model: $MERGED_DIR"
