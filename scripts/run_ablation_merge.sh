#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=sid-ablation-merge
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
    echo "Set VARIANT or submit this script as --array=0-5."
    exit 2
fi
step="${STEP:-}"
if [[ -z "$step" ]]; then
    case "$variant" in
        control|no_validity|no_constrained|exact_only) step=500 ;;
        lora_no_validity|lora_no_constrained) step=200 ;;
        *) echo "Unknown VARIANT='$variant'"; exit 2 ;;
    esac
fi

checkpoint="${artifact_root}/${variant}/step_${step}/raw_actor"
output_dir="${artifact_root}/${variant}/step_${step}/model"
${PYTHON_CMD} -m scripts.merge_fsdp_checkpoint \
    --checkpoint="$checkpoint" \
    --output-dir="$output_dir" \
    --base-model="${BASE_MODEL:-${artifact_root}/stage2_baseline/model}" \
    --mode=auto \
    --use-cpu-init

${PYTHON_CMD} - "$output_dir" <<'PY'
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

path = sys.argv[1]
AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu")
print(f"Verified standalone model at {path}: {model.__class__.__name__}")
PY

sha256sum "$output_dir"/* >"${artifact_root}/${variant}/step_${step}/model.sha256"
