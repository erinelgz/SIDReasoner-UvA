#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=yelp-data-prep
#SBATCH --output=slurm_output/%x-%j.out

# Submit:
#   sbatch scripts/prepare_yelp_data.sh
#
# Override defaults with environment variables:
#   CATEGORY=all MIN_K=10 sbatch scripts/prepare_yelp_data.sh

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"

PYTHON_PREP="${PYTHON_PREP:-python}"
TAR_PATH="${TAR_PATH:?TAR_PATH must be set (e.g. TAR_PATH=/path/to/yelp_dataset.tar sbatch ...)}"
OUT_DIR="${OUT_DIR:-./data/Yelp}"
CATEGORY="${CATEGORY:-Restaurants}"
MIN_K="${MIN_K:-5}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-10}"
MAX_USERS="${MAX_USERS:-10000}"
AMAZON_GENERAL="${AMAZON_GENERAL:-./data/Amazon/general/sampled_data.arrow}"

mkdir -p slurm_output

echo "=========================================="
echo "Yelp data preparation"
echo "  TAR_PATH   : ${TAR_PATH}"
echo "  OUT_DIR    : ${OUT_DIR}"
echo "  CATEGORY   : ${CATEGORY}"
echo "  MIN_K      : ${MIN_K}"
echo "  MAX_SEQ_LEN: ${MAX_SEQ_LEN}"
echo "  MAX_USERS  : ${MAX_USERS}"
echo "=========================================="

"${PYTHON_PREP}" prepare_yelp_data.py \
    --tar_path       "${TAR_PATH}" \
    --out_dir        "${OUT_DIR}" \
    --category       "${CATEGORY}" \
    --min_interactions "${MIN_K}" \
    --max_seq_len    "${MAX_SEQ_LEN}" \
    --max_users      "${MAX_USERS}" \
    --amazon_general_path "${AMAZON_GENERAL}"

echo "Data preparation finished successfully."
