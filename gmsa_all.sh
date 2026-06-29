#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_model="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
train_file="${TRAIN_FILE:-/path/to/train.jsonl}"
test_file="${TEST_FILE:-/path/to/test.jsonl}"

merge_size="${MERGE_SIZE:-16}"
merge_sizes="${MERGE_SIZES:-16,32}"
is_random="${IS_RANDOM:-False}"
use_transform_layer="${USE_TRANSFORM_LAYER:-True}"
fusion_layers="${NUM_MEM_FUSION_LAYERS:-1}"

lora_r="${LORA_R:-128}"
lora_alpha="${LORA_ALPHA:-32}"
lora_dropout="${LORA_DROPOUT:-0.05}"
batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
eval_batch_size="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
accumulation_steps="${GRAD_ACC_STEPS:-1}"

ae_steps="${AE_MAX_STEPS:-5000}"
ae_lr="${AE_LR:-1e-4}"
ae_output="${AE_OUTPUT_DIR:-${script_dir}/output/gmsa_ae_qwen3-4b}"
ft_steps="${FT_MAX_STEPS:-10000}"
ft_lr="${FT_LR:-1e-5}"
ft_output="${FT_OUTPUT_DIR:-${script_dir}/output/gmsa_ft_qwen3-4b}"

deepspeed_config="${DEEPSPEED_CONFIG:-${script_dir}/ds_config.json}"
nproc="${NPROC_PER_NODE:-2}"
master_port="${MASTER_PORT:-29503}"
model_max_length="${MODEL_MAX_LENGTH:-280000}"

common_options=(
  --model_name_or_path "${base_model}"
  --train_file "${train_file}"
  --test_file "${test_file}"
  --merge_size "${merge_size}"
  --merge_sizes "${merge_sizes}"
  --is_random "${is_random}"
  --use_transform_layer "${use_transform_layer}"
  --num_mem_fusion_layers "${fusion_layers}"
  --lora_r "${lora_r}"
  --lora_alpha "${lora_alpha}"
  --lora_dropout "${lora_dropout}"
  --per_device_train_batch_size "${batch_size}"
  --per_device_eval_batch_size "${eval_batch_size}"
  --gradient_accumulation_steps "${accumulation_steps}"
  --model_max_length "${model_max_length}"
  --bf16 True
  --remove_unused_columns False
  --encoder_layers 8
)
if [[ -n "${deepspeed_config}" ]]; then
  common_options+=(--deepspeed "${deepspeed_config}")
fi

launch=(torchrun --nproc_per_node "${nproc}" --master_port "${master_port}")
mkdir -p "${ae_output}" "${ft_output}"

echo "[1/2] GMSA autoencoding"
"${launch[@]}" "${script_dir}/autoencoding.py" \
  "${common_options[@]}" \
  --output_dir "${ae_output}" \
  --max_steps "${ae_steps}" \
  --learning_rate "${ae_lr}"

echo "[2/2] GMSA instruction finetuning"
"${launch[@]}" "${script_dir}/instruction_finetune.py" \
  "${common_options[@]}" \
  --restore_from "${AE_CHECKPOINT:-${ae_output}}" \
  --output_dir "${ft_output}" \
  --max_steps "${ft_steps}" \
  --learning_rate "${ft_lr}"

echo "AE output: ${ae_output}"
echo "Finetune output: ${ft_output}"
