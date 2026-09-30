"""CPU unit tests for the FedSA and exact FRLoRA state transitions."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import torch

from federated_learning.pflalign import PFLAlignOptimizer


from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import LlamaConfig, LlamaForCausalLM

from federated_learning.fed_global import get_fedsa_local_state, global_aggregate
from federated_learning.frlora import initialize_frlora


class FederatedAlgorithmTests(unittest.TestCase):
    def test_pflalign_optimizer_matches_algorithm_one_step(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        name = "adapter.lora_A.weight"
        optimizer = PFLAlignOptimizer(
            [("adapter.lora_A.default.weight", parameter)], lr=0.2, beta=0.9,
            epsilon=1e-12, local_steps=5,
            delta={name: torch.tensor([0.4])}, v={name: torch.zeros(1)},
            preconditioner={name: torch.zeros(1)},
        )
        parameter.grad = torch.tensor([2.0])
        optimizer.step()
        m, v = torch.tensor([0.2]), torch.tensor([0.4])
        alpha = 1 - 0.1 * torch.tensor([4.0]) / (v + 1e-12)
        preconditioner = alpha * 0 + 0.1 * m.square() / (v + 1e-12)
        gamma = 0.5 - 0.5 * torch.erf(
            m.abs() / torch.sqrt(2 * (v - m.square()) + 1e-12)
        ) * torch.sign(-m * torch.tensor([0.4]))
        expected = 1.0 - 0.2 * preconditioner * 2.0 - gamma * 0.4 / 5
        self.assertTrue(torch.allclose(parameter, expected))
        saved_v, saved_p = optimizer.persistent_state()
        self.assertTrue(torch.allclose(saved_v[name], v))
        self.assertTrue(torch.allclose(saved_p[name], preconditioner))

    def test_scaffold_reset_aggregates_trained_adapter_and_updates_controls(self):
        initial = {"layer.lora_A.weight": torch.tensor([[1.0, 2.0]])}
        local_states = [
            {"layer.lora_A.weight": torch.tensor([[3.0, 4.0]])},
            {"layer.lora_A.weight": torch.tensor([[5.0, 6.0]])},
        ]
        global_control = {key: torch.zeros_like(value) for key, value in initial.items()}
        control_deltas = [
            {key: torch.ones_like(value) * amount for key, value in initial.items()}
            for amount in (2.0, 4.0)
        ]
        args = SimpleNamespace(fed_alg="scaffold_reset", num_clients=2)
        reset_state, updated_control = global_aggregate(
            args,
            copy.deepcopy(initial),
            local_states,
            [1, 1],
            [0, 1],
            0,
            auxiliary_info=(global_control, control_deltas),
        )
        self.assertTrue(torch.equal(reset_state["layer.lora_A.weight"], torch.tensor([[4.0, 5.0]])))
        self.assertTrue(torch.equal(updated_control["layer.lora_A.weight"], torch.tensor([[3.0, 3.0]])))

    def test_fedsa_aggregates_only_a_and_restores_private_b(self):
        global_state = {
            "layer.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "layer.lora_B.weight": torch.tensor([[0.0], [0.0]]),
        }
        local_states = [
            {
                "layer.lora_A.weight": torch.tensor([[3.0, 4.0]]),
                "layer.lora_B.weight": torch.tensor([[10.0], [11.0]]),
            },
            {
                "layer.lora_A.weight": torch.tensor([[5.0, 8.0]]),
                "layer.lora_B.weight": torch.tensor([[20.0], [21.0]]),
            },
        ]
        args = SimpleNamespace(fed_alg="fedsa", aggregation_weighting="sample")
        aggregated, _ = global_aggregate(
            args, copy.deepcopy(global_state), local_states, [1, 3], [0, 1], 0
        )
        self.assertTrue(torch.equal(aggregated["layer.lora_A.weight"], torch.tensor([[4.5, 7.0]])))
        self.assertTrue(torch.equal(aggregated["layer.lora_B.weight"], global_state["layer.lora_B.weight"]))
        client_state = get_fedsa_local_state(aggregated, local_states[1])
        self.assertTrue(torch.equal(client_state["layer.lora_A.weight"], aggregated["layer.lora_A.weight"]))
        self.assertTrue(torch.equal(client_state["layer.lora_B.weight"], local_states[1]["layer.lora_B.weight"]))

    def test_frlora_folds_residual_and_resets_factors(self):
        base = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        )
        model = get_peft_model(
            base,
            LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], task_type="CAUSAL_LM"),
        )
        initial = copy.deepcopy(get_peft_model_state_dict(model))
        state = initialize_frlora(model, initial, lora_alpha=8, lora_rank=4)
        set_peft_model_state_dict(model, state.initial_adapter_state)
        stem, (_, b_key, a_key) = next(iter(state.target_pairs.items()))
        target = dict(model.named_modules())[state.target_pairs[stem][0]]
        base_weight = target.get_base_layer().weight
        effective_before = base_weight.detach().clone() + state.scale * (
            state.initial_adapter_state[b_key] @ state.initial_adapter_state[a_key]
        )

        local_states = [copy.deepcopy(state.initial_adapter_state) for _ in range(2)]
        local_states[0][b_key] += 0.01
        local_states[1][a_key] += 0.02
        args = SimpleNamespace(fed_alg="frlora", aggregation_weighting="sample")
        reset_state, _ = global_aggregate(
            args,
            copy.deepcopy(state.initial_adapter_state),
            local_states,
            [1, 3],
            [0, 1],
            0,
            model=model,
            frlora_state=state,
        )
        self.assertLess(state.max_initial_relative_error, 1e-5)
        self.assertTrue(all(torch.allclose(reset_state[key], state.initial_adapter_state[key]) for key in reset_state))

        b_bar = local_states[0][b_key] * 0.25 + local_states[1][b_key] * 0.75
        a_bar = local_states[0][a_key] * 0.25 + local_states[1][a_key] * 0.75
        expected = effective_before + state.scale * (
            b_bar @ a_bar - state.initial_adapter_state[b_key] @ state.initial_adapter_state[a_key]
        )
        effective_after = base_weight.detach() + state.scale * (
            reset_state[b_key] @ reset_state[a_key]
        )
        self.assertTrue(torch.allclose(effective_after, expected, atol=2e-5, rtol=2e-5))

    def test_frlora_scaffold_folds_residual_and_updates_server_control(self):
        base = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        )
        model = get_peft_model(
            base,
            LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], task_type="CAUSAL_LM"),
        )
        initial = copy.deepcopy(get_peft_model_state_dict(model))
        state = initialize_frlora(model, initial, lora_alpha=8, lora_rank=4)
        local_states = [copy.deepcopy(state.initial_adapter_state) for _ in range(2)]
        local_states[0][next(iter(local_states[0]))] += 0.01
        global_control = {key: torch.zeros_like(value) for key, value in initial.items()}
        control_deltas = [copy.deepcopy(global_control) for _ in range(2)]
        for value in control_deltas[0].values():
            value.add_(2.0)
        for value in control_deltas[1].values():
            value.add_(4.0)

        args = SimpleNamespace(fed_alg="frlora_scaffold", aggregation_weighting="sample", num_clients=2)
        reset_state, updated_control = global_aggregate(
            args,
            copy.deepcopy(state.initial_adapter_state),
            local_states,
            [1, 1],
            [0, 1],
            0,
            auxiliary_info=(global_control, control_deltas),
            model=model,
            frlora_state=state,
        )
        self.assertTrue(all(torch.allclose(value, torch.full_like(value, 3.0)) for value in updated_control.values()))
        self.assertTrue(all(torch.allclose(reset_state[key], state.initial_adapter_state[key]) for key in reset_state))


if __name__ == "__main__":
    unittest.main()
