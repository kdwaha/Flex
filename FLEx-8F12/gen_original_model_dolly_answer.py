import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, GenerationConfig
# from bitsandbytes import BitsAndBytesConfig
from utils.dolly_eval import *
from datasets import load_dataset
import argparse
from peft import PeftModel
from utils import *
DEVICE = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="/data/fan/Qwen1.5-MoE-A2.7B")
parser.add_argument("--topk", type=int, default=4)
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--test_data_path", type=str, default="vicuna")
parser.add_argument("--save_path", type=str, default="vicuna")
parser.add_argument("--cuda", type=str, default="cuda:0")
parser.add_argument("--add_side_experts", action="store_true", default=False)
parser.add_argument("--personalized", action="store_true", default=False)
parser.add_argument("--num_side_experts", type=int, default=1)
parser.add_argument("--side_top_k", type=int, default=1)
parser.add_argument("--client_id", type=float, default=0)
parser.add_argument("--len_dataset", type=int, default=50)
parser.add_argument("--num_trials", type=int, default=5)
args = parser.parse_args()


# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True, 
#     bnb_4bit_quant_type="nf4", 
#     bnb_4bit_compute_dtype=torch.bfloat16,  
#     bnb_4bit_use_double_quant=False 
# )

tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)
config = AutoConfig.from_pretrained(args.base_model_path, trust_remote_code=True)
config.num_experts_per_tok = args.topk
config.output_router_logits = False
config.norm_topk_prob = False
# config._attn_implementation = 'flash_attention_2'
print(args.add_side_experts)
if args.add_side_experts :
    config.add_side_experts = args.add_side_experts
    config.num_side_experts = args.num_side_experts
    config.side_top_k = args.side_top_k
    config.side_activation = 1

base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path,
    config=config,
    # quantization_config=quantization_config,  
    device_map=args.cuda,  
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)

base_model.generation_config = GenerationConfig.from_pretrained(args.base_model_path)
base_model.generation_config.pad_token_id = base_model.generation_config.eos_token_id
if args.lora_path :
    print("Use LoRA weight.")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_path,
        torch_dtype=torch.bfloat16,  
        device_map=args.cuda  
    )
else :
    model = base_model
    
if args.add_side_experts :
    model = set_side_expert_dict(args.client_id, model, args)

generate_dolloy_answer(model, tokenizer, args.test_data_path, args.save_path, args.cuda, args) 
# classification_test = "/root/autodl-tmp/OpenFedLLM/dolly_split/classification_test.json"
# classification_test_result_path = './moe_results/classification_test_result.json'
# generate_dolloy_answer(model, tokenizer, classification_test, classification_test_result_path) 

# closed_qa_test = "/root/autodl-tmp/OpenFedLLM/dolly_split/closed_qa_test.json"
# closed_qa_test_result_path = './moe_results/closed_qa_test_result.json'
# generate_dolloy_answer(model, tokenizer, closed_qa_test, closed_qa_test_result_path) 

# information_extraction_test = "/root/autodl-tmp/OpenFedLLM/dolly_split/information_extraction_test.json.json"
# information_extraction_test_result_path = './moe_results/information_extraction_test_result.json'
# generate_dolloy_answer(model, tokenizer, information_extraction_test, information_extraction_test_result_path) 