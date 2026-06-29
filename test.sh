#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test_file="${TEST_FILE:-/path/to/test.jsonl}"
restore_from="${RESTORE_FROM:-/path/to/model}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_SAMPLES="${EVAL_SAMPLES:-100}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${script_dir}/eval_result}"

options=(
  --model_name_or_path "${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
  --test_file "${test_file}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --eval_samples "${EVAL_SAMPLES}"
  --eval_output_dir "${EVAL_OUTPUT_DIR}"
  --model_max_length "${MODEL_MAX_LENGTH:-28000}"
  --merge_size "${MERGE_SIZE:-16}"
  --use_transform_layer "${USE_TRANSFORM_LAYER:-True}"
)
if [[ -n "${restore_from}" ]]; then
  options+=(--restore_from "${restore_from}")
fi

python "${script_dir}/ft_inference.py" "${options[@]}"
