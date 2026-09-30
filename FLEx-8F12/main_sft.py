"""Federated supervised instruction tuning with LoRA.

Supported primary algorithms are standard FedAvg-LoRA, FedSA-LoRA (shared A /
private B), and FRLoRA (residual folding into the frozen base weights).  The
clients are simulated sequentially in one process, but their data and adapter
states remain logically separate.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from trl import SFTTrainer
from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from config import get_config, get_model_config, get_training_args, save_config
from federated_learning import (
    get_auxiliary_dict,
    get_clients_this_round,
    get_dataset_this_round,
    get_fed_local_sft_trainer,
    get_proxy_dict,
    global_aggregate,
    split_dataset,
)
from federated_learning.fed_global import get_fedsa_local_state
from federated_learning.frlora import initialize_frlora, save_merged_frlora_model
from federated_learning.split_dataset import split_train_eval_datasets
from evaluation.instruction_metrics import evaluate_instruction_generations
from evaluation.federated_diagnostics import FederatedDiagnostics, completion_batches, token_metrics, diagnostic_state
from utils import cosine_learning_rate, get_dataset, get_formatting_prompts_func, process_sft_dataset
from utils.completion_collator import EOSPreservingCompletionCollator


def _save_json(path: Path, value) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)


def _client_task_name(dataset) -> str:
    if "category" not in dataset.column_names or not len(dataset):
        return "iid"
    categories = dataset.unique("category")
    return ", ".join(map(str, categories[:5]))


def _response_template_token_ids(tokenizer, formatting_prompts_func, response_template):
    """Get a completion marker sequence valid in a rendered prompt context.

    SentencePiece tokenizers often prepend a standalone whitespace token when
    encoding a marker by itself.  That token does not occur after an instruction
    in the rendered SFT prompt, causing every label to be masked.  Select the
    longest suffix that is actually present in a representative formatted text.
    """

    raw_ids = tokenizer.encode(response_template, add_special_tokens=False)
    probe_text = formatting_prompts_func(
        {"instruction": ["response-marker probe"], "response": ["probe answer"]}
    )[0]
    rendered_ids = tokenizer.encode(probe_text, add_special_tokens=True)
    for start in range(len(raw_ids)):
        candidate = raw_ids[start:]
        if len(candidate) < 2:
            continue
        if any(
            rendered_ids[index : index + len(candidate)] == candidate
            for index in range(len(rendered_ids) - len(candidate) + 1)
        ):
            return candidate
    raise ValueError(
        f"Could not find the response marker `{response_template}` in a rendered prompt."
    )


def _last_subsequence_end(token_ids, marker_ids):
    """Return the end index of the last marker occurrence, or -1."""

    end = -1
    for index in range(len(token_ids) - len(marker_ids) + 1):
        if token_ids[index : index + len(marker_ids)] == marker_ids:
            end = index + len(marker_ids)
    return end


def _keep_completion_examples(dataset, tokenizer, formatting_prompts_func, marker_ids, max_length, label):
    """Drop truncations whose response marker/target would receive no loss."""

    def has_completion(batch):
        texts = formatting_prompts_func(batch)
        tokenized = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        marker_ends = [_last_subsequence_end(ids, marker_ids) for ids in tokenized["input_ids"]]
        return [end >= 0 and end < len(ids) for ids, end in zip(tokenized["input_ids"], marker_ends)]

    filtered = dataset.filter(has_completion, batched=True, desc=f"Checking completion labels for {label}")
    removed = len(dataset) - len(filtered)
    if removed:
        print(f"Dropped {removed}/{len(dataset)} {label} example(s) with a truncated completion marker.")
    return filtered


def _generation_options(script_args):
    """Build deterministic, prompt-only generation settings for client tests."""

    return {
        "template_name": script_args.template,
        "batch_size": script_args.generation_eval_batch_size,
        "max_new_tokens": script_args.generation_max_new_tokens,
        "max_prompt_length": script_args.max_length,
    }


def _aggregate_generation_records(records):
    """Summarize client-task ROUGE-L/BLEU without mixing task assignments."""

    valid = [
        record
        for record in records
        if record.get("examples", 0)
        and record.get("rougeL_f1") is not None
        and record.get("bleu4_pct") is not None
    ]
    if not valid:
        return {"macro_average": None, "example_weighted_average": None}

    metrics = ("rougeL_f1", "bleu4_pct")
    macro_average = {
        metric: sum(float(record[metric]) for record in valid) / len(valid)
        for metric in metrics
    }
    total_examples = sum(int(record["examples"]) for record in valid)
    example_weighted_average = {
        metric: sum(int(record["examples"]) * float(record[metric]) for record in valid)
        / total_examples
        for metric in metrics
    }
    loss_records = [record for record in valid if record.get("eval_loss") is not None]
    if len(loss_records) == len(valid):
        macro_average["eval_loss"] = sum(float(record["eval_loss"]) for record in valid) / len(valid)
        macro_average["perplexity"] = sum(float(record["perplexity"]) for record in valid) / len(valid)
        weighted_loss = sum(
            int(record["examples"]) * float(record["eval_loss"]) for record in valid
        ) / total_examples
        example_weighted_average["eval_loss"] = weighted_loss
        example_weighted_average["perplexity"] = math.exp(weighted_loss)
    return {
        "macro_average": macro_average,
        "example_weighted_average": example_weighted_average,
    }


def _post_local_generation_summary(records):
    """Keep all pre-aggregation measurements and expose each client's latest one."""

    latest_by_client = {}
    for record in records:
        client = int(record["client"])
        if client not in latest_by_client or record["round"] > latest_by_client[client]["round"]:
            latest_by_client[client] = record
    latest_per_client = [latest_by_client[client] for client in sorted(latest_by_client)]
    return {
        "phase": "post_local_train_pre_aggregation",
        "per_client_per_round": records,
        "latest_per_client": latest_per_client,
        **_aggregate_generation_records(latest_per_client),
    }


def _evaluate_post_local_client(
    *,
    model,
    tokenizer,
    client: int,
    round_idx: int,
    heldout_dataset,
    script_args,
    formatting_prompts_func,
    data_collator,
    output_dir: Path,
):
    """Evaluate the client model immediately after its local train call.

    This must run before server aggregation.  In particular, FRLoRA folds the
    averaged residual into the base model and resets its factors at aggregation,
    so the local post-train state cannot be reconstructed later.
    """

    eval_args = get_training_args(script_args, script_args.learning_rate)
    eval_args.output_dir = str(output_dir / "post_local_evaluation" / f"client_{client}")
    eval_args.per_device_eval_batch_size = script_args.batch_size
    eval_args.eval_strategy = "no"
    eval_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=eval_args,
        eval_dataset=heldout_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )
    eval_loss = float(eval_trainer.evaluate()["eval_loss"])
    generation_metrics = evaluate_instruction_generations(
        model,
        tokenizer,
        heldout_dataset,
        **_generation_options(script_args),
    )
    return {
        "round": round_idx + 1,
        "client": client,
        "task": _client_task_name(heldout_dataset),
        "phase": "post_local_train_pre_aggregation",
        "eval_loss": eval_loss,
        "perplexity": math.exp(eval_loss),
        **generation_metrics,
    }


def _evaluate_final_models(
    *,
    model,
    tokenizer,
    eval_datasets,
    global_dict,
    local_dict_list,
    fed_alg,
    script_args,
    formatting_prompts_func,
    data_collator,
    output_dir: Path,
):
    """Compute task-wise held-out metrics after the final round.

    FedSA is personalized by design, so each client uses global A plus its own
    B.  FedAvg and FRLoRA use one global model over every client holdout.
    Prompt-only ROUGE-L/BLEU generation is opt-in because it is substantially
    slower than teacher-forced loss evaluation.
    """

    records = []
    for client, dataset in enumerate(eval_datasets):
        if dataset is None or len(dataset) == 0:
            continue
        if fed_alg == "fedsa":
            set_peft_model_state_dict(model, get_fedsa_local_state(global_dict, local_dict_list[client]))
        else:
            set_peft_model_state_dict(model, global_dict)

        batches = completion_batches(tokenizer, formatting_prompts_func, data_collator,
                                     dataset, script_args.max_length)
        with diagnostic_state(model):
            metrics = token_metrics(model, batches)
        record = {
            "client": client,
            "task": _client_task_name(dataset),
            "examples": len(dataset),
            "eval_loss": metrics["test_loss"],
            **metrics,
        }
        if script_args.generation_eval:
            record.update(
                evaluate_instruction_generations(
                    model,
                    tokenizer,
                    dataset,
                    **_generation_options(script_args),
                )
            )
        records.append(record)

    if not records:
        return {"per_client": [], "weighted_eval_loss": None}
    total = sum(record["target_tokens"] for record in records)
    weighted_loss = sum(record["nll_sum"] for record in records) / total
    result = {
        "per_client": records,
        "weighted_eval_loss": weighted_loss,
        "weighted_perplexity": math.exp(weighted_loss),
        "weighting": "completion_tokens",
    }
    if script_args.generation_eval:
        result["generation_metrics"] = _aggregate_generation_records(records)
    return result


def _save_final_artifact(model, tokenizer, *, fed_alg, global_dict, local_dict_list, output_dir: Path) -> None:
    """Persist an inference-correct final artifact for each FL method."""

    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    if fed_alg in {"frlora", "frlora_scaffold"}:
        # FRLoRA has accumulated residuals in the frozen base.  Only a merged
        # full checkpoint is reconstructable without special bookkeeping.
        save_merged_frlora_model(model, str(final_dir), tokenizer)
    elif fed_alg == "fedsa":
        # A single global LoRA does not exist for FedSA: B is private.  Store
        # the server A and every client B so personalized inference is exact.
        torch.save(
            {"global_adapter_state": global_dict, "client_adapter_states": local_dict_list},
            final_dir / "fedsa_personalized_states.pt",
        )
        tokenizer.save_pretrained(final_dir)
    else:
        set_peft_model_state_dict(model, global_dict)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)


def main() -> None:
    script_args, fed_args, peft_config = get_config()
    if not script_args.use_peft or peft_config is None:
        raise ValueError("Federated LoRA algorithms require `--use_peft`.")
    if script_args.add_side_experts:
        raise NotImplementedError(
            "The released side-expert path is model-specific and incomplete. "
            "Use dense LoRA targets for reproducible FedAvg/FedSA/FRLoRA experiments."
        )

    set_seed(script_args.seed)
    save_config(script_args, fed_args)
    output_dir = Path(script_args.output_dir)
    print(script_args, fed_args)

    # ----- Dataset normalization and deterministic client/task partition -----
    dataset = get_dataset(
        script_args.dataset_name,
        script_args.local_data_dir,
        dataset_config_name=script_args.dataset_config_name,
    )
    dataset = process_sft_dataset(
        script_args.dataset_name,
        dataset,
        script_args.dataset_sample,
        task_column=script_args.task_column,
    )
    if script_args.test_data_dir:
        # A separately supplied test split is never passed to local training.
        # `get_dataset(..., split="test")` also honors a saved DatasetDict or a
        # local directory containing distinct train/test subdirectories.
        test_dataset = get_dataset(
            script_args.dataset_name,
            script_args.test_data_dir,
            dataset_config_name=script_args.dataset_config_name,
            split="test",
        )
        test_dataset = process_sft_dataset(
            script_args.dataset_name,
            test_dataset,
            script_args.test_dataset_sample,
            task_column=script_args.task_column,
        )
        local_datasets = split_dataset(fed_args, script_args, dataset)
        local_eval_datasets = split_dataset(fed_args, script_args, test_dataset)
        evaluation_source = "external_test_data"
    else:
        # The deterministic record-level partition is disjoint: each held-out
        # example is removed from the corresponding client's local train set.
        local_datasets, local_eval_datasets = split_train_eval_datasets(
            fed_args, script_args, dataset, eval_fraction=fed_args.eval_fraction
        )
        evaluation_source = "deterministic_client_heldout_split"
    # ----- Model / tokenizer -----
    device_map, quantization_config, torch_dtype = get_model_config(script_args)
    model_config = AutoConfig.from_pretrained(
        script_args.model_name_or_path, trust_remote_code=script_args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        config=model_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        **({"attn_implementation": "eager"} if script_args.diagnostics_every else {}),
    )
    model.generation_config = GenerationConfig.from_pretrained(script_args.model_name_or_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    if script_args.load_in_8bit or script_args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=script_args.gradient_checkpointing
        )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    if script_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, use_fast=False, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    formatting_prompts_func, response_template = get_formatting_prompts_func(
        script_args.template, tokenizer.eos_token
    )
    response_template_ids = _response_template_token_ids(
        tokenizer, formatting_prompts_func, response_template
    )
    data_collator = EOSPreservingCompletionCollator(response_template_ids, tokenizer=tokenizer)
    _save_json(output_dir / "loss_spec.json", {
        "completion_loss_spec": data_collator.loss_spec,
        "prompt_and_padding_ignored": True,
        "truncation": "EOS is scored only when present within max_length; never appended to a cut response.",
    })

    # Filter only examples whose response section survives max-length
    # truncation; otherwise TRL masks the entire instance and silently trains
    # on fewer examples than the recorded client partition.
    local_datasets = [
        _keep_completion_examples(
            dataset,
            tokenizer,
            formatting_prompts_func,
            response_template_ids,
            script_args.max_length,
            f"client {client} train",
        )
        for client, dataset in enumerate(local_datasets)
    ]
    local_eval_datasets = [
        _keep_completion_examples(
            dataset,
            tokenizer,
            formatting_prompts_func,
            response_template_ids,
            script_args.max_length,
            f"client {client} held-out",
        )
        for client, dataset in enumerate(local_eval_datasets)
    ]
    empty_train_clients = [client for client, dataset in enumerate(local_datasets) if len(dataset) == 0]
    if empty_train_clients:
        raise ValueError(
            "No completion-bearing training examples remain after tokenization for "
            f"client(s) {empty_train_clients}. Increase --max_length or select shorter tasks."
        )
    sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
    print(f"Client train samples: {sample_num_list}")
    print(f"Client test samples: {[len(dataset) for dataset in local_eval_datasets]}")
    _save_json(
        output_dir / "client_partition.json",
        [
            {
                "client": client,
                "task": _client_task_name(local_datasets[client]),
                "train_examples": len(local_datasets[client]),
                "test_examples": len(local_eval_datasets[client]),
                "test_source": evaluation_source,
            }
            for client in range(fed_args.num_clients)
        ],
    )

    if script_args.baseline_eval_only:
        # Standard PEFT initialization has B=0, so the adapter has no effect.
        assert all(torch.count_nonzero(v).item() == 0 for k, v in
                   get_peft_model_state_dict(model).items() if k.endswith('.lora_B.weight'))
        baseline = FederatedDiagnostics(output_dir, script_args, tokenizer, formatting_prompts_func,
                                        data_collator, local_eval_datasets, 'untrained', 0)
        for client in list(range(fed_args.num_clients)) + [-1]:
            baseline.evaluate(model, 0, client, measure_hessian=False)
        _save_json(output_dir / 'metrics.json', {'algorithm': 'untrained', 'rounds': 0,
                    'optimizer_steps': 0, 'evaluation': baseline.records})
        return

    # ----- Federated state initialization -----
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))
    frlora_state = None
    if fed_args.fed_alg in {"frlora", "frlora_scaffold"}:
        frlora_state = initialize_frlora(
            model,
            global_dict,
            lora_alpha=script_args.peft_lora_alpha,
            lora_rank=script_args.peft_lora_r,
        )
        global_dict = copy.deepcopy(frlora_state.initial_adapter_state)
        set_peft_model_state_dict(model, global_dict)
        print(
            "FRLoRA initialized in the principal singular space; maximum initial "
            f"effective-weight relative error={frlora_state.max_initial_relative_error:.3e}"
        )

    local_dict_list = [copy.deepcopy(global_dict) for _ in range(fed_args.num_clients)]
    pflalign_delta = pflalign_v = pflalign_preconditioner = None
    if fed_args.fed_alg == "pflalign":
        zero_state = {key: torch.zeros_like(value, device="cpu") for key, value in global_dict.items()}
        pflalign_delta = [copy.deepcopy(zero_state) for _ in range(fed_args.num_clients)]
        pflalign_v = [copy.deepcopy(zero_state) for _ in range(fed_args.num_clients)]
        pflalign_preconditioner = [copy.deepcopy(zero_state) for _ in range(fed_args.num_clients)]
    initial_adapter_dict = copy.deepcopy(global_dict)
    diagnostics = FederatedDiagnostics(output_dir, script_args, tokenizer, formatting_prompts_func,
                                       data_collator, local_eval_datasets, fed_args.fed_alg, fed_args.num_rounds) if script_args.diagnostics_every else None
    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)
    training_loss = [[] for _ in range(fed_args.num_clients)]
    post_local_generation_records = []

    # ----- Federated training -----
    for round_idx in tqdm(range(fed_args.num_rounds), desc="federated rounds"):
        clients_this_round = get_clients_this_round(fed_args, round_idx)
        previous_global = copy.deepcopy(global_dict) if diagnostics else None
        evaluate_post_local_this_round = bool(script_args.post_local_generation_eval) and (
            not script_args.post_local_generation_eval_final_round_only
            or round_idx == fed_args.num_rounds - 1
        )
        for client in range(fed_args.num_clients):
            if client not in clients_this_round:
                training_loss[client].append(-1.0)
                continue

            local_start = initial_adapter_dict if fed_args.fed_alg == "scaffold_reset" else global_dict
            if fed_args.fed_alg == "pflalign":
                if fed_args.pflalign_variant in {"no_personalization", "sgd"}:
                    # Remove the entire personalization block, not just its correction.
                    for value in pflalign_delta[client].values():
                        value.zero_()
                local_start = {
                    key: global_dict[key] + pflalign_delta[client][key].to(global_dict[key].device)
                    for key in global_dict
                }
            if fed_args.fed_alg == "fedsa":
                set_peft_model_state_dict(model, get_fedsa_local_state(global_dict, local_dict_list[client]))
            else:
                set_peft_model_state_dict(model, local_start)

            sub_dataset = get_dataset_this_round(
                local_datasets[client], round_idx, fed_args, script_args, client_id=client
            )
            category = _client_task_name(sub_dataset)
            print(f">> ==================== Round {round_idx + 1}: client {client} - {category} ====================")
            round_client_dir = output_dir / f"client_{client}_round_{round_idx + 1}"
            script_args.output_dir = str(round_client_dir)
            new_lr = script_args.learning_rate if script_args.round_lr_schedule == "constant" else cosine_learning_rate(
                round_idx, fed_args.num_rounds, script_args.learning_rate, 1e-6
            )
            training_args = get_training_args(script_args, new_lr)
            if diagnostics:
                training_args.logging_strategy = "epoch"
                training_args.save_strategy = "no"
            pflalign_state = None
            if fed_args.fed_alg == "pflalign":
                steps_per_epoch = math.ceil(
                    len(sub_dataset) / (script_args.batch_size * script_args.gradient_accumulation_steps)
                )
                local_steps = script_args.max_steps if script_args.max_steps > 0 else math.ceil(
                    steps_per_epoch * script_args.num_train_epochs
                )
                training_args.max_grad_norm = 0.0
                pflalign_state = {
                    "delta": pflalign_delta[client],
                    "v": pflalign_v[client],
                    "P": pflalign_preconditioner[client],
                    "local_steps": local_steps,
                    "variant": fed_args.pflalign_variant,
                }

            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=local_start,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
                pflalign_state=pflalign_state,
            )
            results = trainer.train()
            training_loss[client].append(float(results.training_loss))
            if fed_args.fed_alg in {"scaffold", "frlora_scaffold", "scaffold_reset"}:
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))
            if fed_args.fed_alg == "pflalign":
                pflalign_v[client], pflalign_preconditioner[client] = trainer.get_persistent_state()
                pflalign_delta[client] = {
                    key: (local_dict_list[client][key] - global_dict[key]).detach().cpu().clone()
                    for key in global_dict
                }
            if diagnostics:
                diagnostics.local(model, client, round_idx + 1, trainer, local_dict_list[client],
                                  previous_global, local_start, auxiliary_model_list[client], new_lr)
            if evaluate_post_local_this_round and len(local_eval_datasets[client]):
                # Deliberately pre-aggregation: this is the individual client's
                # model after its own task update, not a server-averaged model.
                post_local_generation_records.append(
                    _evaluate_post_local_client(
                        model=model,
                        tokenizer=tokenizer,
                        client=client,
                        round_idx=round_idx,
                        heldout_dataset=local_eval_datasets[client],
                        script_args=script_args,
                        formatting_prompts_func=formatting_prompts_func,
                        data_collator=data_collator,
                        output_dir=output_dir,
                    )
                )

        global_dict, global_auxiliary = global_aggregate(
            fed_args,
            global_dict,
            local_dict_list,
            sample_num_list,
            clients_this_round,
            round_idx,
            proxy_dict=proxy_dict,
            opt_proxy_dict=opt_proxy_dict,
            auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
            peft_config=peft_config,
            model=model,
            frlora_state=frlora_state,
        )
        set_peft_model_state_dict(model, global_dict)
        if diagnostics:
            # FedSA has private B factors: explicitly evaluate an averaged-B
            # global proxy, restoring server state afterwards.
            if fed_args.fed_alg == "fedsa":
                proxy = copy.deepcopy(global_dict)
                for key in proxy:
                    if key.endswith(".lora_B.weight"):
                        proxy[key] = sum(local_dict_list[i][key] * sample_num_list[i]
                                         for i in range(fed_args.num_clients)) / sum(sample_num_list)
                set_peft_model_state_dict(model, proxy)
            diagnostics.server(model, round_idx + 1, previous_global, global_dict,
                               local_dict_list, clients_this_round, global_auxiliary)
            set_peft_model_state_dict(model, global_dict)
        np.save(output_dir / "training_loss.npy", np.array(training_loss, dtype=np.float32))
        if script_args.post_local_generation_eval:
            _save_json(
                output_dir / "post_local_personalized_metrics.json",
                _post_local_generation_summary(post_local_generation_records),
            )

    # ----- Held-out evaluation and final artifacts -----
    evaluation = _evaluate_final_models(
        model=model,
        tokenizer=tokenizer,
        eval_datasets=local_eval_datasets,
        global_dict=global_dict,
        local_dict_list=local_dict_list,
        fed_alg=fed_args.fed_alg,
        script_args=script_args,
        formatting_prompts_func=formatting_prompts_func,
        data_collator=data_collator,
        output_dir=output_dir,
    )
    experiment_summary = {
        "algorithm": fed_args.fed_alg,
        "rounds": fed_args.num_rounds,
        "clients": fed_args.num_clients,
        "sampled_clients": fed_args.sample_clients,
        "aggregation_weighting": fed_args.aggregation_weighting,
        "test_data": {
            "source": evaluation_source,
            "used_for_local_training": False,
            "record_level_disjoint_split": evaluation_source == "deterministic_client_heldout_split",
        },
        "evaluation": evaluation,
    }
    if script_args.generation_eval or script_args.post_local_generation_eval:
        experiment_summary["generation_metric_spec"] = {
            "decoding": "greedy prompt-only generation",
            "rougeL_f1": "mean token-level LCS F1 in [0, 1]",
            "bleu4_pct": "smoothed corpus BLEU-4 percentage in [0, 100]",
            "perplexity": "exp(completion-only teacher-forced cross-entropy)",
        }
    if script_args.post_local_generation_eval:
        experiment_summary["post_local_personalized_evaluation"] = _post_local_generation_summary(
            post_local_generation_records
        )
    _save_json(output_dir / "metrics.json", experiment_summary)
    print(json.dumps(experiment_summary, indent=2))
    if script_args.save_final_model:
        _save_final_artifact(
            model,
            tokenizer,
            fed_alg=fed_args.fed_alg,
            global_dict=global_dict,
            local_dict_list=local_dict_list,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
