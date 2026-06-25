#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=sid-adaptive-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=slurm_output/%x-%A_%a.out

set -euo pipefail
set -x

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_DIR"
source ./scripts/snellius_env.sh

artifact_root="${ARTIFACT_ROOT:-/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner}"
result_root="${RESULT_ROOT:-${artifact_root}/adaptive_reasoning_20260613}"
paper_stage3="${PAPER_STAGE3_MODEL:-/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/Office_Products_stage3_rl_Qwen3-1.7B_seed123_global_step_1500/actor_merged}"
lora_model="${LORA_MODEL:-${artifact_root}/office_products_ablation_20260612/lora_no_constrained/step_200/model}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${MODEL_KEY:-}" && -z "${RUN_KIND:-}" ]]; then
    case "$SLURM_ARRAY_TASK_ID" in
        0) model_key=paper_stage3; run_kind=direct ;;
        1) model_key=paper_stage3; run_kind=generated ;;
        2) model_key=lora_no_constrained; run_kind=direct ;;
        3) model_key=lora_no_constrained; run_kind=generated ;;
        *) echo "Smoke array task must be in 0-3"; exit 2 ;;
    esac
else
    model_key="${MODEL_KEY:-paper_stage3}"
    run_kind="${RUN_KIND:-direct}"
fi

case "$model_key" in
    paper_stage3) base_model="${BASE_MODEL:-$paper_stage3}" ;;
    lora_no_constrained) base_model="${BASE_MODEL:-$lora_model}" ;;
    *) echo "MODEL_KEY must be paper_stage3 or lora_no_constrained"; exit 2 ;;
esac

profile="${PROFILE:-smoke20}"
case "$profile" in
    smoke20)
        max_examples="${MAX_EXAMPLES:-20}"
        reasoning_batch_size="${REASONING_BATCH_SIZE:-4}"
        sid_batch_size="${SID_BATCH_SIZE:-8}"
        ;;
    benchmark100)
        max_examples="${MAX_EXAMPLES:-100}"
        reasoning_batch_size="${REASONING_BATCH_SIZE:-8}"
        sid_batch_size="${SID_BATCH_SIZE:-8}"
        ;;
    validation)
        max_examples="${MAX_EXAMPLES:--1}"
        reasoning_batch_size="${REASONING_BATCH_SIZE:-8}"
        sid_batch_size="${SID_BATCH_SIZE:-8}"
        ;;
    *)
        echo "PROFILE must be smoke20, benchmark100, or validation"
        exit 2
        ;;
esac

split="${SPLIT:-validation}"
allow_test=()
if [[ "$split" == "test" ]]; then
    if [[ "${ALLOW_TEST:-false}" != "true" ]]; then
        echo "Refusing official test evaluation without ALLOW_TEST=true"
        exit 2
    fi
    allow_test=(--allow-test)
fi

reasoning_mode=direct
reasoning_samples=1
temperature=0.0
external_transform=none
truncate_fraction=1.0
result_name=direct

case "$run_kind" in
    direct) ;;
    generated)
        reasoning_mode=generated
        result_name=generated_s1
        ;;
    shuffle)
        reasoning_mode=external
        external_transform=shuffle
        result_name=shuffled
        ;;
    truncate25|truncate50|truncate75)
        reasoning_mode=external
        external_transform=truncate
        truncate_fraction="0.${run_kind#truncate}"
        result_name="$run_kind"
        ;;
    self2|self4)
        reasoning_mode=generated
        reasoning_samples="${run_kind#self}"
        temperature="${TEMPERATURE:-0.7}"
        result_name="$run_kind"
        ;;
    *)
        echo "RUN_KIND must be direct, generated, shuffle, truncate25, truncate50, truncate75, self2, or self4"
        exit 2
        ;;
esac

evaluation_dir="${result_root}/${model_key}/${split}/${profile}"
mkdir -p "$evaluation_dir"
external_args=()
if [[ "$reasoning_mode" == "external" ]]; then
    external_source="${EXTERNAL_REASONING_FILE:-${evaluation_dir}/generated_s1.json}"
    external_args=(
        --external-reasoning-file "$external_source"
        --external-transform "$external_transform"
        --truncate-fraction "$truncate_fraction"
    )
fi

result_json="${evaluation_dir}/${result_name}.json"
metrics_json="${evaluation_dir}/${result_name}.metrics.json"

${PYTHON_CMD} evaluate_adaptive_reasoning.py \
    --base-model "$base_model" \
    --reasoning-mode "$reasoning_mode" \
    --reasoning-samples "$reasoning_samples" \
    --save-sequence-scores \
    --max-examples "$max_examples" \
    --split "$split" \
    "${allow_test[@]}" \
    "${external_args[@]}" \
    --result-json "$result_json" \
    --metrics-json "$metrics_json" \
    --index-file "./data/Amazon/index/Office_Products.index.json" \
    --info-file "./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt" \
    --num-beams "${NUM_BEAMS:-10}" \
    --batch-size "$sid_batch_size" \
    --reasoning-batch-size "$reasoning_batch_size" \
    --max-reasoning-tokens "${MAX_REASONING_TOKENS:-1024}" \
    --max-sid-tokens "${MAX_SID_TOKENS:-16}" \
    --temperature "$temperature" \
    --top-p "${TOP_P:-0.9}" \
    --seed "${SEED:-42}"

${PYTHON_CMD} - "$metrics_json" "$reasoning_mode" "$max_examples" <<'PY'
import json
import sys

metrics_path, reasoning_mode, expected = sys.argv[1:]
metrics = json.load(open(metrics_path))
if metrics.get("errors") != 0:
    raise SystemExit(f"Evaluation has errors: {metrics}")
if int(expected) > 0 and metrics.get("total") != int(expected):
    raise SystemExit(f"Expected {expected} examples, got {metrics.get('total')}")
if metrics.get("catalog_valid@1") != 1.0:
    raise SystemExit(f"Constrained SID ranking produced invalid top-one output: {metrics}")
if reasoning_mode == "generated" and metrics.get("generated_reasoning_close_rate", 0.0) < 0.95:
    raise SystemExit(f"Fewer than 95% of generated rationales closed correctly: {metrics}")
PY
