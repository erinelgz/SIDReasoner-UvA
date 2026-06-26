#!/bin/bash

set -euo pipefail
set -x

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source ./scripts/snellius_environment.sh

readonly RESULT_ROOT="/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner/final_office_test_20260613"
readonly OUTPUT_DIR="${PROJECT_DIR}/experiments/final_office_test_20260613"
readonly FROZEN_MANIFEST="${OUTPUT_DIR}/frozen_manifest.json"

${PYTHON_CMD} analyze_final_office_test.py \
    --result-root "$RESULT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --frozen-manifest "$FROZEN_MANIFEST" \
    --bootstrap-samples 10000 \
    --seed 42
