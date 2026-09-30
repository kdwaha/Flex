#!/usr/bin/env bash
# Comparable task-client FLAN experiment.  Prepare DATA via
# tools/download_instruction_subset.py --dataset flan first.
set -euo pipefail

GPU="${GPU:-0}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA="${DATA:-data/flan_tasks_4x600.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/flan_task_baselines}"
PYTHON_BIN="${PYTHON_BIN:-python}"

common=(
  --model_name_or_path "$MODEL"
  --dataset_name flan --local_data_dir "$DATA" --dataset_sample 2400
  --split_strategy task --num_clients 4 --sample_clients 4
  --category_list cb_10templates copa_10templates rte_10templates cola_10templates
  --num_rounds 20 --max_steps 10 --batch_size 2 --gradient_accumulation_steps 4 --max_length 512
  --learning_rate 2e-4 --peft_lora_r 8 --peft_lora_alpha 16
  --target_modules q_proj k_proj v_proj o_proj --use_peft --torch_dtype bfloat16
  --eval_fraction 0.1 --seed 2025
  --save_strategy no --no_save_final_model
)

for algorithm in fedavg fedsa frlora; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" main_sft.py \
    "${common[@]}" --fed_alg "$algorithm" --output_dir "$OUTPUT_ROOT/$algorithm"
done
