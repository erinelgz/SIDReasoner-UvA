#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=sid-final-test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
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

readonly TEST_FILE="${PROJECT_DIR}/data/Amazon/test/Office_Products_5_2016-10-2018-11.csv"
readonly TEST_SHA256="c1967034e197a7837a8962f2556efaabc12eb5bb84e1fdc79cd0accbce71add5"
readonly PAPER_MODEL="/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/Office_Products_stage3_rl_Qwen3-1.7B_seed123_global_step_1500/actor_merged"
readonly PAPER_MODEL_SHA256="4e6c09841795173ce36dae3a912196e7719efe2fefad5e2c4b34047a65c589e2"
readonly LORA_MODEL="/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner/office_products_ablation_20260612/lora_no_constrained/step_200/model"
readonly LORA_MODEL_SHA256="01df94ef9777c5140ab63cf1d4f7983e022204fb7cf51ffa5d5ea4cabcaadcaa"
readonly RESULT_ROOT="/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner/final_office_test_20260613"

case "${SLURM_ARRAY_TASK_ID:?Submit this script as --array=0-3}" in
    0) model_key="paper_stage3"; run_kind="direct"; base_model="$PAPER_MODEL"; model_sha256="$PAPER_MODEL_SHA256" ;;
    1) model_key="paper_stage3"; run_kind="generated"; base_model="$PAPER_MODEL"; model_sha256="$PAPER_MODEL_SHA256" ;;
    2) model_key="lora_no_constrained"; run_kind="direct"; base_model="$LORA_MODEL"; model_sha256="$LORA_MODEL_SHA256" ;;
    3) model_key="lora_no_constrained"; run_kind="generated"; base_model="$LORA_MODEL"; model_sha256="$LORA_MODEL_SHA256" ;;
    *) echo "SLURM_ARRAY_TASK_ID must be 0, 1, 2, or 3"; exit 2 ;;
esac

test_actual="$(sha256sum "$TEST_FILE" | awk '{print $1}')"
model_actual="$(sha256sum "${base_model}/model.safetensors" | awk '{print $1}')"
[[ "$test_actual" == "$TEST_SHA256" ]] || { echo "Unexpected test checksum: $test_actual"; exit 3; }
[[ "$model_actual" == "$model_sha256" ]] || { echo "Unexpected model checksum: $model_actual"; exit 3; }

result_dir="${RESULT_ROOT}/${model_key}/${run_kind}"
result_json="${result_dir}/result.json"
metrics_json="${result_dir}/metrics.json"
job_json="${result_dir}/job.json"
mkdir -p "$result_dir"

if [[ -e "$result_json" || -e "$metrics_json" || -e "$job_json" ]]; then
    echo "Refusing to overwrite frozen final-test output in ${result_dir}"
    exit 4
fi

reasoning_mode="direct"
if [[ "$run_kind" == "generated" ]]; then
    reasoning_mode="generated"
fi

${PYTHON_CMD} evaluate_adaptive_reasoning.py \
    --base-model "$base_model" \
    --reasoning-mode "$reasoning_mode" \
    --reasoning-samples 1 \
    --save-sequence-scores \
    --max-examples -1 \
    --split test \
    --allow-test \
    --data-path "$TEST_FILE" \
    --result-json "$result_json" \
    --metrics-json "$metrics_json" \
    --index-file "${PROJECT_DIR}/data/Amazon/index/Office_Products.index.json" \
    --info-file "${PROJECT_DIR}/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt" \
    --num-beams 10 \
    --batch-size 8 \
    --reasoning-batch-size 8 \
    --max-input-tokens 2048 \
    --max-reasoning-tokens 1024 \
    --max-sid-tokens 16 \
    --temperature 0.0 \
    --top-p 0.9 \
    --length-penalty 0.0 \
    --seed 42

${PYTHON_CMD} - "$metrics_json" "$job_json" "$model_key" "$run_kind" "$base_model" "$model_sha256" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

metrics_path, job_path, model_key, run_kind, base_model, model_sha256 = sys.argv[1:]
metrics = json.loads(Path(metrics_path).read_text())
expected_sha256 = "c1967034e197a7837a8962f2556efaabc12eb5bb84e1fdc79cd0accbce71add5"

failures = []
if metrics.get("total") != 4866 or metrics.get("evaluated") != 4866:
    failures.append(f"expected 4866 examples, got total={metrics.get('total')} evaluated={metrics.get('evaluated')}")
if metrics.get("errors") != 0:
    failures.append(f"errors={metrics.get('errors')}")
if metrics.get("catalog_valid@1") != 1.0:
    failures.append(f"catalog_valid@1={metrics.get('catalog_valid@1')}")
if metrics.get("data_sha256") != expected_sha256:
    failures.append(f"data_sha256={metrics.get('data_sha256')}")
if run_kind == "generated" and metrics.get("generated_reasoning_close_rate", 0.0) < 0.95:
    failures.append(f"generated_reasoning_close_rate={metrics.get('generated_reasoning_close_rate')}")
if failures:
    raise SystemExit("; ".join(failures))

job = {
    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "hostname": socket.gethostname(),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "model_key": model_key,
    "run_kind": run_kind,
    "base_model": base_model,
    "model_sha256": model_sha256,
    "metrics_path": str(Path(metrics_path).resolve()),
}
Path(job_path).write_text(json.dumps(job, indent=2) + "\n")
PY
