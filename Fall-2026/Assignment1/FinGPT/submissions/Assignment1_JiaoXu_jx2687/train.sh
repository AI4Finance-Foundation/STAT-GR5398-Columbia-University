#!/usr/bin/env bash
set -euo pipefail
: "${FINGPT_PYTHON:?Set isolated Python}"
: "${FINGPT_MODEL_PATH:?Set pinned authorized model snapshot}"
cd "$(dirname "$0")/cloud_runs/llama31_8b"
"$FINGPT_PYTHON" train_lora.py --model "$FINGPT_MODEL_PATH" --arch llama3 --run_name llama31_8b --quant8 --offline --max_length 2048 --epochs 1
