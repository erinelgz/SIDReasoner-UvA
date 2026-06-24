#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=sid-yelp-eval-think
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH --chdir=/gpfs/home2/scur1223/SIDReasoner-UvA

set -euo pipefail

source ./scripts/snellius_env.sh

CATEGORY="${CATEGORY:-Yelp_Restaurants}"
DATASET="${DATASET:-Yelp_Restaurants_5core}"
TEST_FILE="${TEST_FILE:-./data/Yelp/test/${DATASET}_pos3_dedup.csv}"
INFO_FILE="${INFO_FILE:-./data/Yelp/info/${DATASET}.txt}"
ITEM_FILE="${ITEM_FILE:-./data/Yelp/index/Yelp_Restaurants.item.json}"
INDEX_FILE="${INDEX_FILE:-./data/Yelp/index/Yelp_Restaurants.index.json}"
EXP_NAME="${EXP_NAME:-./checkpoints/RecRL_Reasoning/Yelp_Restaurants_stage3_rl_Qwen3-1.7B/global_step_1870/actor_merged}"
CUDA_LIST="${CUDA_LIST:-0 1}"
CUDA_LIST_CSV="${CUDA_LIST_CSV:-0,1}"

dir1=$(basename "$(dirname "$EXP_NAME")")
dir2=$(basename "$EXP_NAME")
exp_name_clean="${dir1}__${dir2}"

echo "Processing category: ${CATEGORY} with model: ${exp_name_clean} (THINK MODE)"

if [[ ! -f "${TEST_FILE}" ]]; then
    echo "Error: Test file not found: ${TEST_FILE}"
    exit 1
fi
if [[ ! -f "${INFO_FILE}" ]]; then
    echo "Error: Info file not found: ${INFO_FILE}"
    exit 1
fi

temp_dir="./temp/${CATEGORY}-think-${exp_name_clean}"
mkdir -p "${temp_dir}" slurm_output

echo "Splitting test data..."
${PYTHON_CMD} ./split.py --input_path "${TEST_FILE}" --output_path "${temp_dir}" --cuda_list "${CUDA_LIST_CSV}"

echo "Starting parallel evaluation..."
for i in ${CUDA_LIST}
do
    if [[ -f "${temp_dir}/${i}.csv" ]]; then
        echo "Starting evaluation on GPU ${i}"
        CUDA_VISIBLE_DEVICES=${i} ${PYTHON_CMD} -u ./evaluate_Qwen3_think.py \
            --base_model "${EXP_NAME}" \
            --info_file "${INFO_FILE}" \
            --category "${CATEGORY}" \
            --test_data_path "${temp_dir}/${i}.csv" \
            --item_file "${ITEM_FILE}" \
            --index_file "${INDEX_FILE}" \
            --result_json_data "${temp_dir}/${i}.json" \
            --batch_size 4 \
            --num_beams 10 \
            --max_new_tokens 1024 \
            --length_penalty 0.0 &
    fi
done
wait
echo "All GPU evaluations complete."

output_dir="./results/${CATEGORY}/${exp_name_clean}-think"
mkdir -p "${output_dir}"

actual_cuda_list=""
for gpu in ${CUDA_LIST}; do
    if [[ -f "${temp_dir}/${gpu}.json" ]]; then
        actual_cuda_list="${actual_cuda_list}${gpu},"
    fi
done
actual_cuda_list="${actual_cuda_list%,}"

echo "Merging results from GPUs: ${actual_cuda_list}"
${PYTHON_CMD} ./merge.py \
    --input_path "${temp_dir}" \
    --output_path "${output_dir}/final_result_${CATEGORY}.json" \
    --cuda_list "${actual_cuda_list}"

if [[ ! -f "${output_dir}/final_result_${CATEGORY}.json" ]]; then
    echo "Error: Result merging failed"
    exit 1
fi

echo "Calculating metrics..."
${PYTHON_CMD} ./calc.py \
    --path "${output_dir}/final_result_${CATEGORY}.json" \
    --item_path "${INFO_FILE}"

echo "Completed. Results saved to: ${output_dir}/final_result_${CATEGORY}.json"
