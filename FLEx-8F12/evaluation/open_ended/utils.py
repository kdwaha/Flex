import math
from safetensors.torch import load_file

import random
from typing import List

def set_side_expert_dict(model, args):
    if args.add_side_experts :
        side_expert_weight = load_file('/home/liuf/pruning_weight/alpaca_r1.safetensors', device="cpu")
        print(f"Load Weight from /home/liuf/pruning_weight/alpaca_r1.safetensors")
        # if args.client == 0 :
        #     side_expert_weight = load_file('/home/liuf/pruning_weight/closedqa_r1.safetensors', device="cpu")
        #     print(f"Load Weight from /home/liuf/pruning_weight/closedqa_r1.safetensors")
        # elif args.client == 1 :
        #     side_expert_weight = load_file('/home/liuf/pruning_weight/info_r1.safetensors', device="cpu")
        #     print(f"Load Weight from /home/liuf/pruning_weight/info_r1.safetensors")
        # elif args.client == 2 :
        #     side_expert_weight = load_file('/home/liuf/pruning_weight/classification_r1.safetensors', device="cpu")
        #     print(f"Load Weight from /home/liuf/pruning_weight/classification_r1.safetensors")
        # elif args.client == 3 :
        #     side_expert_weight = load_file('/home/liuf/pruning_weight/summarization_r1.safetensors', device="cpu")
        #     print(f"Load Weight from /home/liuf/pruning_weight/summarization_r1.safetensors")
    else :
        print("NO CATEGORY! USE NoPer Weight Instead!")
        side_expert_weight = load_file('/home/liuf/pruning_weight/c4_r1.safetensors', device="cpu")
        print(f"Load Weight from /home/liuf/pruning_weight/c4_r1.safetensors")
    target_state_dict_subset = {}
    for loaded_key, weight_tensor in side_expert_weight.items():
        target_key = None
        if 'side_gate.weight' in loaded_key:
            target_key = loaded_key.replace('.side_gate.weight', '.side_gate.base_layer.weight')
        # Map side_experts projections to their base_layer equivalents
        elif 'side_experts.down_proj.weight' in loaded_key:
            target_key = loaded_key.replace('.side_experts.down_proj.weight', '.side_experts.down_proj.base_layer.weight')
        elif 'side_experts.up_proj.weight' in loaded_key:
            target_key = loaded_key.replace('.side_experts.up_proj.weight', '.side_experts.up_proj.base_layer.weight')
        elif 'side_experts.gate_proj.weight' in loaded_key:
            target_key = loaded_key.replace('.side_experts.gate_proj.weight', '.side_experts.gate_proj.base_layer.weight')

        if target_key:
            target_state_dict_subset[target_key] = weight_tensor

    info = model.base_model.model.load_state_dict(target_state_dict_subset, strict=False)
    assert info.unexpected_keys == [], \
        f"Here are some weight key error:{info.unexpected_keys}."
    print(f"Load Side Experts Weight Successfully.")
    return model

"""
To support TRL supervised fine-tuning. Right now, we need to manually set the template here.
"""

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""

vicuna_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT: {}{}"""

TEMPLATE_DICT = {
    'alpaca': (alpaca_template, '\n### Response:'),
    'vicuna': (vicuna_template, ' ASSISTANT:'),
}