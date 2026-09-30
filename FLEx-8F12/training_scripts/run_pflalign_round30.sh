#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="${OUTPUT_ROOT:-outputs/llama1b_gpu0/round30_constant_final_hessian}"
mkdir -p "$ROOT"
VARIANT="${PFLALIGN_VARIANT:-full}"
METHOD=pflalign
if [[ "$VARIANT" != full ]]; then METHOD="pflalign_$VARIANT"; fi
if [[ -e "$ROOT/$METHOD" || -e "$ROOT/$METHOD.log" ]]; then
    echo "Refusing to overwrite existing experiment: $ROOT/$METHOD" >&2
    exit 1
fi
"$PYTHON_BIN" main_sft.py \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset_name dolly --local_data_dir data/dolly_tasks_4x20.jsonl \
    --dataset_sample 80 --split_strategy task \
    --category_list closed_qa information_extraction classification summarization \
    --num_clients 4 --sample_clients 4 --num_rounds "${ROUNDS:-30}" \
    --max_steps "${MAX_STEPS:--1}" --num_train_epochs 5 --batch_size 4 \
    --gradient_accumulation_steps 1 --max_length 512 \
    --learning_rate 2e-4 --round_lr_schedule constant \
    --peft_lora_r 8 --peft_lora_alpha 16 --target_modules q_proj k_proj v_proj o_proj \
    --use_peft --torch_dtype float32 --no_gradient_checkpointing --eval_fraction 0.2 \
    --save_strategy no --no_save_final_model --diagnostics_every "${EVAL_EVERY:-5}" \
    --generation_max_new_tokens "${GEN_TOKENS:-500}" --generation_eval_batch_size 4 \
    --hessian_lanczos_steps "${LANCZOS_STEPS:-20}" --hessian_probes "${LANCZOS_PROBES:-2}" \
    --hessian_trace_probes "${TRACE_PROBES:-4}" --hessian_every 0 --logging_steps 100 --seed 2025 \
    --pflalign_beta "${PFLALIGN_BETA:-0.9}" --fed_alg pflalign \
    --pflalign_variant "$VARIANT" \
    --output_dir "$ROOT/$METHOD" > "$ROOT/$METHOD.log" 2>&1
"$PYTHON_BIN" tools/plot_federated_diagnostics.py "$ROOT/$METHOD"
"$PYTHON_BIN" tools/build_comparison_report.py "$ROOT"
