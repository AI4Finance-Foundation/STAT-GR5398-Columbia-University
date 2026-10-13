#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$(basename "$SCRIPT_DIR")" == "source_code" ]]; then
  PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
else
  PROJECT_DIR="$SCRIPT_DIR"
fi

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
MODEL_KEY="${MODEL_KEY:-qwen}"

case "$MODEL_KEY" in
  qwen)
    DEFAULT_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    DEFAULT_MODEL_FAMILY="qwen2.5"
    DEFAULT_RUN_NAME="qwen_baseline"
    ;;
  llama)
    DEFAULT_MODEL_NAME="meta-llama/Llama-3.1-8B"
    DEFAULT_MODEL_FAMILY="llama3"
    DEFAULT_RUN_NAME="llama_baseline"
    ;;
  deepseek)
    DEFAULT_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    DEFAULT_MODEL_FAMILY="deepseek"
    DEFAULT_RUN_NAME="deepseek_baseline"
    ;;
  *)
    echo "MODEL_KEY must be qwen, llama, or deepseek; received: $MODEL_KEY" >&2
    exit 2
    ;;
esac

MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
MODEL_FAMILY="${MODEL_FAMILY:-$DEFAULT_MODEL_FAMILY}"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
DATASET="${DATASET:-FinGPT/fingpt-forecaster-dow30-202305-202405}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/finetuned_models/$RUN_NAME}"
REPORT_TO="${REPORT_TO:-tensorboard}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SCHEDULER="${SCHEDULER:-constant}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
SEED="${SEED:-42}"

TRAIN_ARGS=(
  train_lora.py
  --model-name "$MODEL_NAME"
  --model-family "$MODEL_FAMILY"
  --dataset "$DATASET"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --max-length "$MAX_LENGTH"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --num-epochs "$NUM_EPOCHS"
  --warmup-ratio "$WARMUP_RATIO"
  --scheduler "$SCHEDULER"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --seed "$SEED"
  --precision auto
  --gradient-checkpointing
  --logging-steps 10
  --eval-strategy no
  --save-strategy epoch
  --save-total-limit 2
  --report-to "$REPORT_TO"
)

if [[ "$USE_DEEPSPEED" == "1" ]]; then
  if ! command -v deepspeed >/dev/null 2>&1; then
    echo "DeepSpeed executable not found. Install deepspeed or set USE_DEEPSPEED=0." >&2
    exit 1
  fi
  deepspeed --num_gpus "$NUM_GPUS" "${TRAIN_ARGS[@]}" \
    --quantization none \
    --optim adamw_torch \
    --ds-config config.json
else
  "$PYTHON_BIN" "${TRAIN_ARGS[@]}" \
    --quantization 4bit \
    --optim paged_adamw_8bit
fi
