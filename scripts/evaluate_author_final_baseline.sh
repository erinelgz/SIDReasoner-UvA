#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=sid-author-final-test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/%x-%A_%a.out

# Evaluate only the author Stage-3 model. Existing reproduced Stage-3 and LoRA
# test outputs remain read-only inputs to the later paired analysis.

set -euo pipefail
set -x

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    project_dir="${SLURM_SUBMIT_DIR}"
else
    project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$project_dir"
source ./scripts/snellius_environment.sh

artifact_root="${ARTIFACT_ROOT:-/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner/author_final_baseline_test_20260623}"
model_path="${AUTHOR_STAGE3_MODEL:-${artifact_root}/author_stage3_model}"
test_file="${project_dir}/data/Amazon/test/Office_Products_5_2016-10-2018-11.csv"
test_sha256="c1967034e197a7837a8962f2556efaabc12eb5bb84e1fdc79cd0accbce71add5"

profile="${PROFILE:-test_full}"
case "$profile" in
    smoke20)
        max_examples=20
        result_root="${RESULT_ROOT:-${artifact_root}/results/test_smoke20/author_stage3}"
        ;;
    test_full)
        max_examples=-1
        result_root="${RESULT_ROOT:-${artifact_root}/results/test/author_stage3}"
        ;;
    *)
        echo "PROFILE must be smoke20 or test_full" >&2
        exit 2
        ;;
esac

case "${SLURM_ARRAY_TASK_ID:?Submit this script as --array=0-1}" in
    0) run_kind="direct" ;;
    1) run_kind="generated" ;;
    *) echo "SLURM_ARRAY_TASK_ID must be 0 or 1" >&2; exit 2 ;;
esac

[[ -f "${model_path}/config.json" && -f "${model_path}/model.safetensors" ]] || {
    echo "Author model is unavailable at ${model_path}. Run scripts/fetch_author_office_rl_checkpoint.sh first." >&2
    exit 3
}
actual_test_sha256="$(sha256sum "$test_file" | awk '{print $1}')"
[[ "$actual_test_sha256" == "$test_sha256" ]] || {
    echo "Unexpected test checksum: $actual_test_sha256" >&2
    exit 4
}

result_dir="${result_root}/${run_kind}"
result_json="${result_dir}/result.json"
metrics_json="${result_dir}/metrics.json"
job_json="${result_dir}/job.json"
mkdir -p "$result_dir"
if [[ -e "$result_json" || -e "$metrics_json" || -e "$job_json" ]]; then
    echo "Refusing to overwrite author-final test output in ${result_dir}" >&2
    exit 5
fi

reasoning_mode="direct"
if [[ "$run_kind" == "generated" ]]; then
    reasoning_mode="generated"
fi

${PYTHON_CMD} evaluate_adaptive_reasoning.py \
    --base-model "$model_path" \
    --reasoning-mode "$reasoning_mode" \
    --reasoning-samples 1 \
    --save-sequence-scores \
    --max-examples "$max_examples" \
    --split test \
    --allow-test \
    --data-path "$test_file" \
    --result-json "$result_json" \
    --metrics-json "$metrics_json" \
    --index-file "${project_dir}/data/Amazon/index/Office_Products.index.json" \
    --info-file "${project_dir}/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt" \
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

model_sha256="$(sha256sum "${model_path}/model.safetensors" | awk '{print $1}')"
${PYTHON_CMD} - "$metrics_json" "$job_json" "$run_kind" "$model_path" "$model_sha256" "$test_sha256" "$max_examples" "$profile" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

metrics_path, job_path, run_kind, model_path, model_sha256, expected_test_sha256, max_examples, profile = sys.argv[1:]
metrics = json.loads(Path(metrics_path).read_text())
failures = []
expected_examples = 4866 if int(max_examples) < 0 else int(max_examples)
if metrics.get("total") != expected_examples or metrics.get("evaluated") != expected_examples:
    failures.append(
        f"expected {expected_examples} examples, got total={metrics.get('total')} evaluated={metrics.get('evaluated')}"
    )
if metrics.get("errors") != 0:
    failures.append(f"errors={metrics.get('errors')}")
if metrics.get("catalog_valid@1") != 1.0:
    failures.append(f"catalog_valid@1={metrics.get('catalog_valid@1')}")
if metrics.get("data_sha256") != expected_test_sha256:
    failures.append(f"data_sha256={metrics.get('data_sha256')}")
if run_kind == "generated" and metrics.get("generated_reasoning_close_rate", 0.0) < 0.95:
    failures.append(f"generated_reasoning_close_rate={metrics.get('generated_reasoning_close_rate')}")
if failures:
    raise SystemExit("; ".join(failures))

Path(job_path).write_text(
    json.dumps(
        {
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": socket.gethostname(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_key": "author_stage3",
            "run_kind": run_kind,
            "profile": profile,
            "base_model": model_path,
            "model_sha256": model_sha256,
            "metrics_path": str(Path(metrics_path).resolve()),
        },
        indent=2,
    )
    + "\n"
)
PY
