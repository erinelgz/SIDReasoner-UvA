#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "The CUDA 12.4 / Torch 2.6 / vLLM / SGLang dependencies are now managed by pyproject.toml."
echo ""
echo "Use:"
echo "  uv sync"
echo ""
echo "For optional Megatron and TransformerEngine support, use:"
echo "  uv sync --extra megatron"
echo ""
echo "This compatibility wrapper will now run uv for you."

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not available on PATH."
    exit 1
fi

if [[ "${USE_MEGATRON:-0}" == "1" ]]; then
    uv sync --extra megatron
else
    uv sync
fi
