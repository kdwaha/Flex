#!/usr/bin/env bash
# A functional CPU smoke run.  It validates the full SFT path, not model quality.
set -euo pipefail

DATA="${DATA:-data/dolly_tasks_4x20.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-python}"
for algorithm in fedavg fedsa frlora; do
  "$PYTHON_BIN" main_sft.py \
    --model_name_or_path hf-internal-testing/tiny-random-LlamaForCausalLM \
    --dataset_name dolly --local_data_dir "$DATA" --dataset_sample 80 \
    --split_strategy task --category_list closed_qa information_extraction classification summarization \
    --num_clients 4 --sample_clients 4 --num_rounds 1 --max_steps 1 \
    --batch_size 1 --gradient_accumulation_steps 1 --max_length 512 \
    --learning_rate 1e-3 --peft_lora_r 4 --peft_lora_alpha 8 \
    --target_modules q_proj k_proj v_proj o_proj --use_peft \
    --torch_dtype float32 --no_gradient_checkpointing --eval_fraction 0.2 \
    --save_strategy no --no_save_final_model \
    --fed_alg "$algorithm" --output_dir "outputs/cpu_smoke/$algorithm"
done
