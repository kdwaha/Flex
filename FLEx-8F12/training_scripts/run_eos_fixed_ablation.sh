#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="${OUTPUT_ROOT:-outputs/llama1b_gpu0/eos_fixed_ablation_20260921}"
# Refuse to overwrite an existing experiment, including a failed one.
mkdir "$ROOT"
common=(
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0
    --dataset_name dolly --local_data_dir data/dolly_tasks_4x20.jsonl
    --dataset_sample 80 --split_strategy task
    --category_list closed_qa information_extraction classification summarization
    --num_clients 4 --sample_clients 4 --num_rounds 30
    --max_steps -1 --num_train_epochs 5 --batch_size 4
    --gradient_accumulation_steps 1 --max_length 512
    --learning_rate 2e-4 --round_lr_schedule constant
    --peft_lora_r 8 --peft_lora_alpha 16 --target_modules q_proj k_proj v_proj o_proj
    --use_peft --torch_dtype float32 --no_gradient_checkpointing --eval_fraction 0.2
    --save_strategy no --no_save_final_model --diagnostics_every 5
    --generation_max_new_tokens 500 --generation_eval_batch_size 4
    --hessian_lanczos_steps 20 --hessian_probes 2 --hessian_trace_probes 4
    --hessian_every 0 --logging_steps 100 --seed 2025
)
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "GPU required"; print(torch.cuda.get_device_name(0))'
for method in untrained fedavg fedsa frlora frlora_scaffold scaffold scaffold_reset pflalign \
    pflalign_no_preconditioner pflalign_no_correction pflalign_constant_gamma \
    pflalign_hard_gamma pflalign_no_personalization pflalign_sgd; do
    extra=()
    algorithm="$method"
    if [[ "$method" == untrained ]]; then
        algorithm=fedavg
        extra+=(--baseline_eval_only)
    elif [[ "$method" == pflalign_* ]]; then
        algorithm=pflalign
        extra+=(--pflalign_variant "${method#pflalign_}")
    fi
    "$PYTHON_BIN" main_sft.py "${common[@]}" --fed_alg "$algorithm" "${extra[@]}" \
        --output_dir "$ROOT/$method" > "$ROOT/$method.log" 2>&1
    # Only finalized runs are consumed. No checkpoints or intermediate log reads.
    "$PYTHON_BIN" tools/plot_federated_diagnostics.py "$ROOT/$method" > "$ROOT/${method}_plots.log" 2>&1
    "$PYTHON_BIN" tools/build_comparison_report.py "$ROOT" > "$ROOT/report_build.log" 2>&1
    printf 'COMPLETED %s\n' "$method"
done
printf 'SUITE COMPLETE: %s/report/comparison.md\n' "$ROOT"
