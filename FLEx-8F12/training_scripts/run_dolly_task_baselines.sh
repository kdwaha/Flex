#!/usr/bin/env bash
# Run directly comparable FedAvg-LoRA, FedSA-LoRA, and FRLoRA experiments.
# First create DATA with tools/download_instruction_subset.py.
set -euo pipefail

GPU="${GPU:-0}"
# Public Llama-family 1.1B default; override MODEL for a gated Meta Llama checkpoint.
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
DATA="${DATA:-data/dolly_tasks_4x600.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/dolly_task_baselines}"
PYTHON_BIN="${PYTHON_BIN:-python}"

common=(
  --model_name_or_path "$MODEL"
  --dataset_name dolly
  --local_data_dir "$DATA"
  --dataset_sample 2400
  --split_strategy task
  --num_clients 4 --sample_clients 4
  --category_list closed_qa information_extraction classification summarization
  --num_rounds 20 --max_steps 10
  --batch_size 2 --gradient_accumulation_steps 4 --max_length 512
  --learning_rate 2e-4
  --peft_lora_r 8 --peft_lora_alpha 16
  --target_modules q_proj k_proj v_proj o_proj
  --use_peft --torch_dtype bfloat16
  --eval_fraction 0.1 --seed 2025
  --save_strategy no --no_save_final_model
)

for algorithm in fedavg fedsa frlora; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" main_sft.py \
    "${common[@]}" --fed_alg "$algorithm" --output_dir "$OUTPUT_ROOT/$algorithm"
done
