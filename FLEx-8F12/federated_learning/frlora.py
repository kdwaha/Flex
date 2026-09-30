"""Exact FRLoRA utilities for PEFT LoRA models.

FRLoRA (Yan et al., ICLR 2025) is not an alternative way of averaging LoRA
state dictionaries.  It folds the round's *residual low-rank update* into the
frozen base weight, then resets all LoRA factors to the same PiSSA/SVD
initialization for the next round.  The functions in this module keep that
state transition explicit and deliberately reject quantized or unsupported
target modules in the caller.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch


AdapterState = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class FRLoRAState:
    """Fixed initialization and target-module mapping for an FRLoRA run."""

    initial_adapter_state: AdapterState
    # ``stem -> (module_name, lora_B state key, lora_A state key)``.
    target_pairs: Mapping[str, Tuple[str, str, str]]
    scale: float
    max_initial_relative_error: float


def _lora_pairs(adapter_state: Mapping[str, torch.Tensor]) -> Dict[str, Tuple[str, str]]:
    """Return matching PEFT state keys for all LoRA A/B factor pairs.

    PEFT's exported state format normally uses ``...lora_A.weight`` and
    ``...lora_B.weight``.  Keeping this parser in one place makes the error
    readable if a future PEFT version changes the representation.
    """

    pairs: Dict[str, Dict[str, str]] = {}
    for key in adapter_state:
        if key.endswith(".lora_A.weight"):
            pairs.setdefault(key[: -len(".lora_A.weight")], {})["A"] = key
        elif key.endswith(".lora_B.weight"):
            pairs.setdefault(key[: -len(".lora_B.weight")], {})["B"] = key

    missing = [stem for stem, pair in pairs.items() if set(pair) != {"A", "B"}]
    if missing:
        raise ValueError(f"Incomplete LoRA A/B state for target(s): {missing[:3]}")
    if not pairs:
        raise ValueError("FRLoRA requires a PEFT model with LoRA A/B parameters.")
    return {stem: (pair["B"], pair["A"]) for stem, pair in pairs.items()}


def _resolve_target_module(model: torch.nn.Module, state_stem: str) -> Tuple[str, torch.nn.Module]:
    """Map a PEFT adapter-state prefix to its LoRA-wrapped module."""

    modules = dict(model.named_modules())
    candidates = [state_stem]
    for prefix in ("base_model.model.", "base_model.", "model."):
        if state_stem.startswith(prefix):
            candidates.append(state_stem[len(prefix) :])
        else:
            candidates.append(prefix + state_stem)

    for candidate in candidates:
        module = modules.get(candidate)
        if module is not None and hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            return candidate, module

    # PEFT can add one wrapper prefix depending on the model class.  Resolve a
    # unique suffix as a compatibility fallback, but never guess ambiguously.
    matches = [
        (name, module)
        for name, module in modules.items()
        if (name.endswith(state_stem) or state_stem.endswith(name))
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"Could not resolve LoRA target module for state key `{state_stem}`.")
    raise KeyError(f"Ambiguous LoRA target module for `{state_stem}`: {[name for name, _ in matches]}")


def _base_weight(module: torch.nn.Module) -> torch.nn.Parameter:
    """Return the frozen 2-D weight behind a PEFT LoRA target module."""

    base_layer = module.get_base_layer() if hasattr(module, "get_base_layer") else getattr(module, "base_layer", module)
    weight = getattr(base_layer, "weight", None)
    if weight is None or weight.ndim != 2:
        raise TypeError("FRLoRA currently supports only 2-D linear LoRA target weights.")
    return weight


def _relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(torch.finfo(torch.float32).eps)
    return float(torch.linalg.vector_norm((reference - candidate).float()) / denominator)


@torch.no_grad()
def initialize_frlora(
    model: torch.nn.Module,
    adapter_state: Mapping[str, torch.Tensor],
    *,
    lora_alpha: float,
    lora_rank: int,
) -> FRLoRAState:
    """Perform the one-time principal-singular-space initialization in FRLoRA.

    For each LoRA target ``W0``, this writes ``W_hat0 = W0 - s B0 A0`` to
    the frozen base weight and returns an adapter state containing ``B0, A0``.
    The effective weight is therefore unchanged at initialization.
    """

    if lora_rank <= 0 or lora_alpha <= 0:
        raise ValueError("FRLoRA requires positive `peft_lora_r` and `peft_lora_alpha`.")
    scale = float(lora_alpha) / float(lora_rank)
    initial_state: AdapterState = copy.deepcopy(dict(adapter_state))
    pairs = _lora_pairs(initial_state)
    target_pairs: Dict[str, Tuple[str, str, str]] = {}
    max_relative_error = 0.0

    for stem, (b_key, a_key) in pairs.items():
        module_name, module = _resolve_target_module(model, stem)
        weight = _base_weight(module)
        b_shape, a_shape = initial_state[b_key].shape, initial_state[a_key].shape
        if weight.shape != (b_shape[0], a_shape[1]):
            raise ValueError(
                f"FRLoRA target `{stem}` has base shape {tuple(weight.shape)}, but "
                f"LoRA factors imply {(b_shape[0], a_shape[1])}. "
                "Fan-in/fan-out targets are not supported by this exact implementation."
            )
        rank = min(lora_rank, weight.shape[0], weight.shape[1])
        if rank != lora_rank:
            raise ValueError(f"LoRA rank {lora_rank} exceeds a target dimension for `{stem}`.")

        # SVD in fp32 is required because most CUDA kernels do not accept bf16
        # here.  It is intentionally done exactly once, on the pretrained W0.
        # ``Tensor.float()`` returns an alias when the base is already fp32;
        # clone before the in-place W_hat update so the initialization check is
        # a real comparison against immutable W0.
        original = weight.detach().float().clone()
        u, singular_values, vh = torch.linalg.svd(original, full_matrices=False)
        root = torch.sqrt(singular_values[:rank] / scale)
        b0_fp32 = u[:, :rank] * root.unsqueeze(0)
        a0_fp32 = root.unsqueeze(1) * vh[:rank, :]

        b0 = b0_fp32.to(device=initial_state[b_key].device, dtype=initial_state[b_key].dtype)
        a0 = a0_fp32.to(device=initial_state[a_key].device, dtype=initial_state[a_key].dtype)
        initial_state[b_key] = b0
        initial_state[a_key] = a0

        # Keep the effective initial forward weight equal to W0.
        initial_update = (scale * (b0_fp32 @ a0_fp32)).to(device=weight.device, dtype=weight.dtype)
        weight.sub_(initial_update)
        reconstructed = weight.detach() + initial_update
        max_relative_error = max(max_relative_error, _relative_error(original, reconstructed))
        target_pairs[stem] = (module_name, b_key, a_key)

    return FRLoRAState(
        initial_adapter_state=initial_state,
        target_pairs=target_pairs,
        scale=scale,
        max_initial_relative_error=max_relative_error,
    )


def aggregate_adapter_states(
    local_states: Sequence[Mapping[str, torch.Tensor]],
    client_ids: Iterable[int],
    sample_num_list: Sequence[int],
    weighting: str = "sample",
) -> AdapterState:
    """Aggregate adapters, using the same weights for all LoRA factors."""

    selected = list(client_ids)
    if not selected:
        raise ValueError("Cannot aggregate an empty client set.")
    if weighting == "sample":
        total = sum(sample_num_list[client] for client in selected)
        if total <= 0:
            raise ValueError("Selected clients have no training examples.")
        weights = {client: sample_num_list[client] / total for client in selected}
    elif weighting == "uniform":
        weights = {client: 1.0 / len(selected) for client in selected}
    else:
        raise ValueError(f"Unknown aggregation weighting: {weighting}")

    reference = local_states[selected[0]]
    return {
        key: sum(local_states[client][key] * weights[client] for client in selected)
        for key in reference
    }


@torch.no_grad()
def fold_frlora_residual(
    model: torch.nn.Module,
    aggregated_adapter_state: Mapping[str, torch.Tensor],
    state: FRLoRAState,
) -> None:
    """Add one round's FRLoRA residual to every frozen base target weight."""

    modules = dict(model.named_modules())
    for stem, (module_name, b_key, a_key) in state.target_pairs.items():
        module = modules.get(module_name)
        if module is None:
            raise KeyError(f"FRLoRA target module disappeared: `{module_name}` for `{stem}`.")
        weight = _base_weight(module)
        b_round = aggregated_adapter_state[b_key].detach().float()
        a_round = aggregated_adapter_state[a_key].detach().float()
        b0 = state.initial_adapter_state[b_key].detach().float()
        a0 = state.initial_adapter_state[a_key].detach().float()
        residual = state.scale * (b_round @ a_round - b0 @ a0)
        weight.add_(residual.to(device=weight.device, dtype=weight.dtype))


def reset_adapter_state(state: FRLoRAState) -> AdapterState:
    """Return a fresh B0/A0 state; clients must never continue Bbar/Abar."""

    return copy.deepcopy(state.initial_adapter_state)


@torch.no_grad()
def save_merged_frlora_model(model: torch.nn.Module, output_dir: str, tokenizer=None) -> None:
    """Merge the fixed final B0/A0 into the residual-updated base and save it.

    A normal PEFT adapter checkpoint would omit the accumulated base-weight
    residuals, so FRLoRA's final artifact is intentionally a full model.
    This function is only safe after the final training round because merging
    unloads the PEFT wrappers.
    """

    if not hasattr(model, "merge_and_unload"):
        raise TypeError("FRLoRA final save expects a PEFT model with `merge_and_unload`.")
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
