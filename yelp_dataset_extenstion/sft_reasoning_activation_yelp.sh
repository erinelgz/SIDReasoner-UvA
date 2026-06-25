#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=sid-yelp-stage2-activation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH --chdir=/gpfs/home2/scur1223/SIDReasoner-UvA

set -euo pipefail

source ./scripts/snellius_environment.sh
export WANDB_MODE=disabled

CATEGORY="${CATEGORY:-Yelp_Restaurants}"
DATASET="${DATASET:-Yelp_Restaurants_5core}"
TRAIN_FILE="${TRAIN_FILE:-./data/Yelp/train/${DATASET}.csv}"
EVAL_FILE="${EVAL_FILE:-./data/Yelp/valid/${DATASET}.csv}"
TEST_FILE="${TEST_FILE:-./data/Yelp/test/${DATASET}.csv}"
INFO_FILE="${INFO_FILE:-./data/Yelp/info/${DATASET}.txt}"
BASE_MODEL="${BASE_MODEL:-./output_dir/${CATEGORY}_stage1_sft_Qwen3-1.7B/final_checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_dir/${CATEGORY}_stage2_reasoning_activation_Qwen3-1.7B}"
RUN_NAME="${RUN_NAME:-${CATEGORY}_stage2_reasoning_activation_Qwen3-1.7B}"
LOG_FILE="${LOG_FILE:-./logs/${RUN_NAME}.txt}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29520}"

mkdir -p ./logs ./output_dir slurm_output

{
echo "${TRAIN_FILE} ${EVAL_FILE} ${INFO_FILE} ${TEST_FILE}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" ${TORCHRUN_CMD} --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    sft_reasoning_activation.py \
    --base_model "${BASE_MODEL}" \
    --batch_size 1024 \
    --micro_batch_size 8 \
    --train_file "${TRAIN_FILE}" \
    --eval_file "${EVAL_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --wandb_project MiniOneRec \
    --wandb_run_name "${RUN_NAME}" \
    --category "${CATEGORY}" \
    --train_from_scratch False \
    --seed 42 \
    --sid_index_path "./data/Yelp/index/Yelp_Restaurants.index.json" \
    --item_meta_path "./data/Yelp/index/Yelp_Restaurants.item.json" \
    --reasoning_train_file "./data/Yelp/index/Yelp_Restaurants.integrated_narrative.csv" \
    --train_new_token_embeddings_only False \
    "$@"
} > "${LOG_FILE}" 2>&1
