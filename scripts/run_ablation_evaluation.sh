#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=sid-ablation-eval
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

artifact_root="${ARTIFACT_ROOT:-/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner/office_products_ablation_20260612}"
variants=(control no_validity no_constrained exact_only lora_no_validity lora_no_constrained)
if [[ -n "${VARIANT:-}" ]]; then
    variant="$VARIANT"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    variant="${variants[$SLURM_ARRAY_TASK_ID]}"
else
    variant=stage2_baseline
fi
mode="${MODE:-common}"
step="${STEP:-}"

if [[ "$variant" == "stage2_baseline" ]]; then
    model="${BASE_MODEL:-${artifact_root}/stage2_baseline/model}"
    evaluation_dir="${artifact_root}/stage2_baseline/evaluation"
    constrained=True
else
    if [[ -z "$step" ]]; then
        case "$variant" in
            control|no_validity|no_constrained|exact_only) step=500 ;;
            lora_no_validity|lora_no_constrained) step=200 ;;
            *) echo "Unknown VARIANT='$variant'"; exit 2 ;;
        esac
    fi
    model="${BASE_MODEL:-${artifact_root}/${variant}/step_${step}/model}"
    evaluation_dir="${artifact_root}/${variant}/step_${step}/evaluation"
    case "$variant" in
        no_constrained|lora_no_constrained) constrained=False ;;
        *) constrained=True ;;
    esac
fi

mkdir -p "$evaluation_dir"
case "$mode" in
    common)
        constrained=True
        direct=True
        samples=-1
        beams=10
        max_tokens=32
        batch_size=16
        ;;
    native)
        direct=True
        samples=-1
        beams=10
        max_tokens=32
        batch_size=16
        ;;
    reasoning_smoke)
        direct=False
        samples=100
        beams=1
        max_tokens=256
        batch_size=4
        ;;
    *)
        echo "MODE must be common, native, or reasoning_smoke."
        exit 2
        ;;
esac

samples="${NUM_SAMPLES:-$samples}"
beams="${NUM_BEAMS:-$beams}"
max_tokens="${MAX_NEW_TOKENS:-$max_tokens}"
batch_size="${BATCH_SIZE:-$batch_size}"
result_name="${RESULT_NAME:-$mode}"

${PYTHON_CMD} evaluate_verl_constrained.py \
    --base_model="$model" \
    --parquet_path="./data/Amazon/rec_reasoning_verl/Office_Products/test.parquet" \
    --index_file="./data/Amazon/index/Office_Products.index.json" \
    --info_file="./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt" \
    --num_beams="$beams" \
    --max_new_tokens="$max_tokens" \
    --batch_size="$batch_size" \
    --num_samples="$samples" \
    --seed=42 \
    --constrained_decoding="$constrained" \
    --direct_sid_decoding="$direct" \
    --save_full_text="$([[ "$mode" == "reasoning_smoke" ]] && echo True || echo False)" \
    --result_json_data="${evaluation_dir}/${result_name}.json" \
    --metrics_json_data="${evaluation_dir}/${result_name}.metrics.json"

${PYTHON_CMD} - "${evaluation_dir}/${result_name}.metrics.json" "$constrained" "$mode" <<'PY'
import json
import sys

metrics_path, constrained, mode = sys.argv[1:]
with open(metrics_path) as handle:
    metrics = json.load(handle)
if metrics.get("errors") != 0 or metrics.get("evaluated") != metrics.get("total"):
    raise SystemExit(f"Evaluation failed its zero-error gate: {metrics}")
if constrained.lower() == "true" and metrics.get("catalog_valid@1") != 1.0:
    raise SystemExit(f"Constrained decoding produced an invalid SID: {metrics}")
if mode == "reasoning_smoke":
    if metrics.get("reasoning_enclosed@1", 0.0) <= 0.0:
        raise SystemExit(f"No complete <think>...</think> output was observed: {metrics}")
    if metrics.get("reward_parseable@1", 0.0) <= 0.0:
        raise SystemExit(f"No reasoning output could be parsed by the reward function: {metrics}")
PY
