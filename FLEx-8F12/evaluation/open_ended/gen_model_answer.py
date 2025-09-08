import datasets
import argparse
import json
import sys
sys.path.append("../../")
from tqdm import tqdm
import os
import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from utils import set_side_expert_dict
from utils import TEMPLATE_DICT

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="/data/fan/Qwen1.5-MoE-A2.7B")
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--cuda", type=str, default=None)
parser.add_argument("--template", type=str, default="alpaca")
parser.add_argument("--use_vllm", action="store_true")
parser.add_argument("--add_side_experts", action="store_true")
parser.add_argument("--bench_name", type=str, default="vicuna")
args = parser.parse_args()
print(args)

if args.use_vllm and args.lora_path is not None:
    raise ValueError("Cannot use both VLLM and LORA, need to merge the lora and then use VLLM")

template = TEMPLATE_DICT[args.template][0]
print(f">> You are using template: {template}")

# ============= Load dataset =============
if args.bench_name == "alpaca":
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    max_new_tokens = 2048
elif args.bench_name == "vicuna":
    eval_set = datasets.load_dataset("json", data_files="/home/liuf/OpenFedLLM/evaluation/open_ended/data/vicuna/question.jsonl")['train']
    # eval_set = eval_set.rename_column("text", "instruction")
    def rename(example):
        example['instruction'] = example['turns'][0]
        return example
    eval_set = eval_set.map(rename)
    max_new_tokens = 256
elif args.bench_name == "advbench":
    eval_set = datasets.load_dataset("csv", data_files="/home/liuf/OpenFedLLM/evaluation/open_ended/data/advbench/advbench.csv")["train"]
    eval_set = eval_set.rename_column("goal", "instruction")
    eval_set = eval_set.remove_columns(["target"])
    max_new_tokens = 128
else:
    raise ValueError("Invalid benchmark name")

# ============= Extract model name from the path. The name is used for saving results. =============
if args.lora_path:
    pre_str, checkpoint_str = os.path.split(args.lora_path)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    if 'add_Per' in pre_str:
        model_name = f"add_Per_{exp_name}"
    elif 'AVG' in pre_str:
        model_name = f"AVG_{exp_name}"
    elif 'fedavgm' in pre_str:
        model_name = f"fedavgm_{exp_name}"
    elif 'fedadam' in pre_str:
        model_name = f"fedadam_{exp_name}"
    elif 'fedadagrad' in pre_str:
        model_name = f"fedadagrad_{exp_name}"
    elif 'fedprox' in pre_str:
        model_name = f"fedprox_{exp_name}"
    elif 'fedyogi' in pre_str:
        model_name = f"fedyogi_{exp_name}"
    elif 'scaffold' in pre_str:
        model_name = f"scaffold_{exp_name}"
    elif 'local' in pre_str:
        model_name = f"local_{exp_name}"
    else :
        model_name = f"{exp_name}_{checkpoint_id}"

else:
    pre_str, last_str = os.path.split(args.base_model_path)
    if last_str.startswith("full"):                 # if the model is merged as full model
        _, exp_name = os.path.split(pre_str)
        checkpoint_id = last_str.split("-")[-1]
        model_name = f"{exp_name}_{checkpoint_id}"
    else:
        model_name = last_str                       # mainly for base model

# ============= Load previous results if exists =============
result_path = f"/home/liuf/OpenFedLLM/evaluation/open_ended/data/{args.bench_name}/model_answer/{model_name}.json"

if os.path.exists(result_path):
    with open(result_path, "r") as f:
        result_list = json.load(f)
else:
    result_list = []
existing_len = len(result_list)
print(f">> Existing length: {existing_len}")

# ============= Generate responses =============
if args.use_vllm:
    model = LLM(model=args.base_model_path)
    if args.bench_name == "advbench":
        input_list = [template.format(example["instruction"]+'.', "", "")[:-1] for example in eval_set]
    else:
        input_list = [template.format(example["instruction"], "", "")[:-1] for example in eval_set] # TODO: use fastchat conversation
    input_list = input_list[existing_len:]
    print(f">> Example input: {input_list[0]}")
    sampling_params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=max_new_tokens)
    generations = model.generate(input_list, sampling_params)
    generations = [generation.outputs[0].text for generation in generations]

    for i, example in tqdm(enumerate(eval_set)):
        if i < existing_len:
            continue
        example['output'] = generations[i-existing_len]
        example['generator'] = exp_name
        result_list.append(example)
    with open(result_path, "w") as f:
        json.dump(result_list, f, indent=4)

else:
    device = args.cuda
    config = AutoConfig.from_pretrained(args.base_model_path, trust_remote_code=True)
    if args.add_side_experts :
        config.add_side_experts = args.add_side_experts
        config.num_side_experts = 1
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16, config=config, trust_remote_code=True).to(device)
    if args.lora_path is not None:
        model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=torch.bfloat16).to(device)
        if args.add_side_experts :
            model = set_side_expert_dict(model, args)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)
    

    for i, example in tqdm(enumerate(eval_set)):
        if i < existing_len:
            continue
        if args.bench_name == "advbench":
            instruction = template.format(example["instruction"]+'.', "", "")[:-1]
        else:
            instruction = template.format(example["instruction"], "", "")[:-1]      # TODO: use fastchat conversation
        input_ids = tokenizer.encode(instruction, return_tensors="pt").to(device)
        output_ids = model.generate(inputs=input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=1.0, temperature=0.7)
        output_ids = output_ids[0][len(input_ids[0]):]
        result = tokenizer.decode(output_ids, skip_special_tokens=True)
        example['output'] = result
        example['generator'] = model_name

        print(f"\nInput: \n{instruction}")
        print(f"\nOutput: \n{result}")
        print("="*100)
        result_list.append(example)
        with open(result_path, "w") as f:
            json.dump(result_list, f, indent=4)

print(f">> You are using template: {template}")
