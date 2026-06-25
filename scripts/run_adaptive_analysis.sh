#!/bin/bash

set -euo pipefail
set -x

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source ./scripts/snellius_env.sh

artifact_root="${ARTIFACT_ROOT:-/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner}"
result_root="${RESULT_ROOT:-${artifact_root}/adaptive_reasoning_20260613}"
model_key="${MODEL_KEY:-paper_stage3}"
split="${SPLIT:-validation}"
profile="${PROFILE:-validation}"
evaluation_dir="${result_root}/${model_key}/${split}/${profile}"

direct_results="${DIRECT_RESULTS:-${evaluation_dir}/direct.json}"
generated_results="${GENERATED_RESULTS:-${evaluation_dir}/generated_s1.json}"
analysis_dir="${ANALYSIS_DIR:-${evaluation_dir}/analysis}"

optional_args=()
for control in shuffled truncate25 truncate50 truncate75; do
    path="${evaluation_dir}/${control}.json"
    if [[ -f "$path" ]]; then
        optional_args+=(--control "${control}=${path}")
    fi
done
for sample in self2 self4; do
    path="${evaluation_dir}/${sample}.json"
    if [[ -f "$path" ]]; then
        optional_args+=(--self-consistency "${sample}=${path}")
    fi
done

${PYTHON_CMD} analyze_adaptive_reasoning.py \
    --direct-results "$direct_results" \
    --generated-results "$generated_results" \
    --output-dir "$analysis_dir" \
    --calibration-seed "${CALIBRATION_SEED:-42}" \
    --calibration-fraction "${CALIBRATION_FRACTION:-0.5}" \
    --thinking-budgets "${THINKING_BUDGETS:-0.25,0.5,0.75}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
    "${optional_args[@]}"
