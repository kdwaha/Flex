import random
import torch

from .frlora import (
    FRLoRAState,
    aggregate_adapter_states,
    fold_frlora_residual,
    reset_adapter_state,
)


def get_fedsa_local_state(global_dict, client_dict):
    """Combine shared LoRA-A with one client's persistent private LoRA-B.

    FedSA-LoRA trains both factors locally but sends only A to the server.  A
    fresh dictionary is returned so the sequential single-process simulator
    cannot accidentally mutate the saved private B state before training.
    """

    state = {key: value.detach().clone() for key, value in global_dict.items()}
    for key, value in client_dict.items():
        if key.endswith(".lora_B.weight"):
            state[key] = value.detach().clone()
    return state

def get_clients_this_round(fed_args, round):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            random.seed(round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round

def global_aggregate(
    fed_args,
    global_dict,
    local_dict_list,
    sample_num_list,
    clients_this_round,
    round_idx,
    proxy_dict=None,
    opt_proxy_dict=None,
    auxiliary_info=None,
    peft_config=None,
    model=None,
    frlora_state: FRLoRAState | None = None,
):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None

    if fed_args.fed_alg == 'scaffold':
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]) 
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients
    elif fed_args.fed_alg == 'scaffold_reset':
        # Evaluate the trained aggregate; reset happens only at local start.
        for key in global_dict:
            global_dict[key] = sum(
                local_dict_list[client][key] * sample_num_list[client] / sample_this_round
                for client in clients_this_round
            )
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum(
                auxiliary_delta_dict[client][key] for client in clients_this_round
            )
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients
    elif fed_args.fed_alg == 'flexlora':
        # FlexLoRA aggregation
        if peft_config is None:
            raise ValueError("`peft_config` is required for flexlora aggregation.")
        for key in global_dict.keys():
            if key.endswith("lora_B.weight"):
                layer_name = key.replace(".lora_B.weight", "")
                lora_A_key = layer_name + ".lora_A.weight"
                # Compute W_g
                W_g = sum([(sample_num_list[client] / sample_this_round) * (peft_config.lora_alpha / peft_config.r) * local_dict_list[client][key] @ local_dict_list[client][lora_A_key] for client in clients_this_round])
                # Perform SVD
                U, S, V = torch.svd(W_g, some=True)
                r_max = peft_config.r
                global_dict[key] = U[:, :r_max] @ torch.diag(S[:r_max]) / (peft_config.lora_alpha / peft_config.r)
                global_dict[lora_A_key] = V[:, :r_max].T
            else:
                # For other keys, average them
                global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])

    elif fed_args.fed_alg == 'fedsa':
        # FedSA-LoRA shares only the A factor.  B stays permanently local and
        # is restored for each client by main_sft.py before local training.
        for key in global_dict.keys():
            if key.endswith(".lora_A.weight"):
                global_dict[key] = sum(
                    local_dict_list[client][key] * sample_num_list[client] / sample_this_round
                    for client in clients_this_round
                )

    elif fed_args.fed_alg in ['frlora', 'frlora_scaffold']:
        # FRLoRA aggregates B/A for this round, folds their residual product
        # into frozen base weights, then resets adapters to the fixed B0/A0.
        # It must not be reduced to ordinary adapter-state averaging.
        if model is None or frlora_state is None:
            raise ValueError("FRLoRA aggregation requires both `model` and `frlora_state`.")
        round_adapter_state = aggregate_adapter_states(
            local_dict_list,
            clients_this_round,
            sample_num_list,
            weighting=fed_args.aggregation_weighting,
        )
        fold_frlora_residual(model, round_adapter_state, frlora_state)
        global_dict = reset_adapter_state(frlora_state)
        if fed_args.fed_alg == 'frlora_scaffold':
            global_auxiliary, auxiliary_delta_dict = auxiliary_info
            for key in global_auxiliary.keys():
                delta_auxiliary = sum(
                    auxiliary_delta_dict[client][key] for client in clients_this_round
                )
                global_auxiliary[key] += delta_auxiliary / fed_args.num_clients
                
    elif fed_args.fed_alg == 'fedavgm':
        # Momentum-based FedAvg
        for key in global_dict.keys():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            global_dict[key] = global_dict[key] + proxy_dict[key]

    elif fed_args.fed_alg == 'fedadagrad':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            # In paper 'adaptive federated optimization', momentum is not used
            proxy_dict[key] = delta_w
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedyogi':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(param - delta_square)
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedadam':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            opt_proxy_dict[key] = fed_args.fedopt_beta2*param + (1-fed_args.fedopt_beta2)*torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    else:   # Normal dataset-size-based aggregation 
        for key in global_dict.keys():
            if 'side' in key or 'expert' in key or 'gate' in key :
                continue
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    
    return global_dict, global_auxiliary
