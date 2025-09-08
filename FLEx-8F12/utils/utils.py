import math
from safetensors.torch import load_file

import random
from typing import List

def set_side_expert_dict(client, model, args, local_dataset=None):
    if args.personalized and not local_dataset:
        # side_expert_weight = load_file(your_path, device="cpu")
        print("Here Put Your pruning weight.")
    elif args.personalized and local_dataset:
        if 'category' not in local_dataset.column_names:
            print("Warning: No 'category' Column")
            print(f"Load Weight from [Your Weight Path]")
            print(f"Load c4 weight instead")
        else:
            category_counts = {}
            for example in local_dataset:
                category = example['category']
                category_counts[category] = category_counts.get(category, 0) + 1

            if category_counts:
                most_common_category = max(category_counts, key=category_counts.get)
                print(f"The category with the highest proportion in the local dataset is: {most_common_category}")
                # Here for dirichlet distribution.
            else:
                print("Local datasets have no category information")
                print(f"Load c4 weight instead") 
    else :
        # Here for no personalized weight
        print(f"Load c4 weight instead") 
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

def select_random_expert_modules(
    num_layers: int,
    num_experts_per_layer: int,
    expert_linear_names: List[str] = None,
    base_path_template: str = "model.layers.{layer_idx}.mlp.experts.{expert_idx}.{linear_name}"
) -> List[str]:

    if expert_linear_names is None:
        expert_linear_names = ['gate_proj', 'up_proj', 'down_proj'] 

    target_modules_list = []

    for layer_idx in range(num_layers):
        selected_expert_idx = random.randint(0, num_experts_per_layer - 1)

        for linear_name in expert_linear_names:
            module_name = base_path_template.format(
                layer_idx=layer_idx,
                expert_idx=selected_expert_idx,
                linear_name=linear_name
            )
            target_modules_list.append(module_name)

    return target_modules_list

def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr


if __name__ == "__main__":

    # Example usage:
    num_rounds = 300
    initial_lr = 5e-5
    min_lr = 1e-6

    lr_list = []
    for round in range(num_rounds):
        lr = cosine_learning_rate(round, num_rounds, initial_lr, min_lr)
        lr_list.append(lr)
        print(f"Round {round + 1}/{num_rounds}, Learning Rate: {lr:.8f}")
