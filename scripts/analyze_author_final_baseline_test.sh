#!/bin/bash

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
source ./scripts/snellius_env.sh

artifact_root="${ARTIFACT_ROOT:-/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner/author_final_baseline_test_20260623}"
frozen_root="${FROZEN_RESULT_ROOT:-/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner/final_office_test_20260613}"
author_public_data_archive="${AUTHOR_PUBLIC_DATA_ARCHIVE:-${artifact_root}/source_data/Amazon.zip}"
local_data_root="${LOCAL_DATA_ROOT:-${project_dir}/data/Amazon}"

${PYTHON_CMD} analyze_author_final_baseline_test.py \
    --author-result-root "${artifact_root}/results/test/author_stage3" \
    --frozen-result-root "$frozen_root" \
    --provenance "${artifact_root}/provenance.json" \
    --author-public-data-archive "$author_public_data_archive" \
    --local-data-root "$local_data_root" \
    --output-dir "${artifact_root}/analysis" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
    --seed "${SEED:-42}"
