#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=sid-aligned-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output/%x-%j.out

set -euo pipefail
set -x

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_DIR"

source ./scripts/snellius_env.sh

MODE="${MODE:-model}"
BASE_MODEL="${BASE_MODEL:-/home/scur1249/Office_Products_checkpoint/merged}"
PARQUET_PATH="${PARQUET_PATH:-./data/Amazon/rec_reasoning_verl/Office_Products/test.parquet}"
TRAIN_PARQUET="${TRAIN_PARQUET:-./data/Amazon/rec_reasoning_verl/Office_Products/train.parquet}"
INDEX_FILE="${INDEX_FILE:-./data/Amazon/index/Office_Products.index.json}"
INFO_FILE="${INFO_FILE:-./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt}"
RESULT_DIR="${RESULT_DIR:-./temp}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-aligned_${MODE}_$(date +%Y%m%d_%H%M%S)}"
NUM_BEAMS="${NUM_BEAMS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
SEED="${SEED:-42}"
CONSTRAINED_DECODING="${CONSTRAINED_DECODING:-true}"
DIRECT_SID_DECODING="${DIRECT_SID_DECODING:-true}"

mkdir -p "$RESULT_DIR"

if [[ "$MODE" == "model" ]]; then
    ${PYTHON_CMD} evaluate_verl_constrained.py \
        --base_model="${BASE_MODEL}" \
        --parquet_path="${PARQUET_PATH}" \
        --index_file="${INDEX_FILE}" \
        --info_file="${INFO_FILE}" \
        --num_beams="${NUM_BEAMS}" \
        --max_new_tokens="${MAX_NEW_TOKENS}" \
        --batch_size="${BATCH_SIZE}" \
        --num_samples="${NUM_SAMPLES}" \
        --seed="${SEED}" \
        --constrained_decoding="${CONSTRAINED_DECODING}" \
        --direct_sid_decoding="${DIRECT_SID_DECODING}" \
        --result_json_data="${RESULT_DIR}/${EXPERIMENT_NAME}.json"
elif [[ "$MODE" == "baselines" ]]; then
    ${PYTHON_CMD} evaluate_sid_baselines.py \
        --train_parquet="${TRAIN_PARQUET}" \
        --test_parquet="${PARQUET_PATH}" \
        --index_file="${INDEX_FILE}" \
        --info_file="${INFO_FILE}" \
        --num_samples="${NUM_SAMPLES}" \
        --seed="${SEED}" \
        --result_json_data="${RESULT_DIR}/${EXPERIMENT_NAME}.json"
else
    echo "MODE must be either 'model' or 'baselines', got '${MODE}'"
    exit 2
fi
