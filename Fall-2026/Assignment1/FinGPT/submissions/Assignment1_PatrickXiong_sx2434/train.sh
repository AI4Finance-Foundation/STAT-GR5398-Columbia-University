#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -lt 2 ]; then
  echo "Usage: bash train.sh qwen|deepseek|llama /root/shared-nvme/AI4FinRuntime/runs/FRESH_DIR [--execute]"
  exit 2
fi
MODEL="$1"
OUT="$2"
shift 2
exec /root/shared-nvme/AI4FinRuntime/venvs/fingpt/bin/python "$SCRIPT_DIR/train_lora.py" --model "$MODEL" --out "$OUT" "$@"
