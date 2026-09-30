# FLEx

> The maintained federated instruction-tuning path is documented in
> [`doc/federated_instruction_tuning.md`](doc/federated_instruction_tuning.md).
> It includes task-per-client Dolly/FLAN data adapters, FedAvg-LoRA,
> FedSA-LoRA, and the ICLR 2025 FRLoRA implementation.  Use
> `requirements-federated.txt` rather than the historical pinned
> `requirements.txt` for this path.

The maintained path also includes SCAFFOLD/reset, pFLAlign ablations,
EOS-preserving completion loss, and periodic local/global diagnostics.
See the [published results and reproduction guide](results/eos_fixed_20260921/README.md)
for tables, client curves, and Hessian plots. The historical setup below is
retained for reference; use the maintained guide for these experiments.

## Setup

Clone the repo, submodules and install the required packages.

```
git clone https://anonymous.4open.science/r/FLEx-8F12.git
cd FLEx
conda create -n fedllm python=3.10
conda activate fedllm
pip install -r requirements.txt
source setup.sh
```

### Federated Instruction Tuning

The training script is in `training_scripts/run_sft_fedavg_random_experts.sh`.

```
CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --local_data_dir $local_data_dir \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft \
 --output_dir $output_dir \
 --template "alpaca" \
 --split_strategy $split_strategy \
 --category_list "${category_list[@]}" \
 --target_modules "${target_modules[@]}" \
 --add_side_experts $add_side_experts \
 --trust_remote_code $trust_remote_code \
 --personalized $personalized \
 --random_select_experts $random_select_experts
```

Key arguments:

- `add_side_experts`: Use pruning experts
- `personalized`: Use personalized experts
- `category_list`: Your dataset tasks
- `target_modules`: LoRA target_modules
- `random_select_experts`: Random select experts and add lora with them.

This project based on :

**OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning**

*Authors*: Rui Ye, Wenhao Wang, Jingyi Chai, Dihan Li, Zexi Li, Yinda Xu, Yaxin Du, Yanfeng Wang, Siheng Chen

*Conference*: Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '24)

*Year*: 2024

*Pages*: 6137-6147

**Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models** 

*Authors*: Xudong Lu and Qi Liu and Yuhui Xu and Aojun Zhou and Siyuan Huang and Bo Zhang and Junchi Yan and Hongsheng Li

*Conference*: In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics

*Year*: 2024 

*Pages*: 6159-6172
