max_steps=5
num_rounds=100
batch_size=4
gradient_accumulation_steps=1
seq_length=512
num_clients=4
sample_clients=4
lora_r=32
lora_alpha=64   # twice of lora_r
lr=4e-5

local_data_dir=""       
dataset_name="dolly"
dataset_sample=200000
model_name_or_path="/data/fan/Qwen1.5-MoE-A2.7B"
output_dir=
add_side_experts=False # Use FLEx
# personalized=False # Use personalized weight
trust_remote_code=True

gpu=4
fed_alg="fedavg"
split_strategy="no-iid"
category_list=('closed_qa' 'information_extraction' 'classification' 'summarization')
target_modules=('q_proj' 'k_proj' 'v_proj')
random_select_experts=True # random select experts and add LoRA

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