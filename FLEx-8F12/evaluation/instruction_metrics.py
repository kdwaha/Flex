"""Generation metrics for task-wise federated instruction tuning.

The helpers in this module deliberately avoid ``evaluate.load(...)`` so a
post-training evaluation can run offline and does not download metric scripts.
ROUGE-L is reported as an example-mean F1 in ``[0, 1]`` and BLEU is a smoothed
corpus BLEU-4 percentage in ``[0, 100]``.  Both use the same deterministic,
case-insensitive tokenizer, which makes small client-held-out evaluations
reproducible.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
import math
import re
from typing import Any, Optional

import torch

from utils.template import TEMPLATE_DICT


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def metric_tokenize(text: Any) -> list[str]:
    """Tokenize text consistently for the local ROUGE-L and BLEU metrics."""

    return _TOKEN_PATTERN.findall(str(text or "").lower())


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Return the LCS length using only one dynamic-programming row."""

    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: Any, reference: Any) -> float:
    """Compute token-level ROUGE-L F1 for one prediction/reference pair."""

    prediction_tokens = metric_tokenize(prediction)
    reference_tokens = metric_tokenize(reference)
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    lcs = _lcs_length(prediction_tokens, reference_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _ngram_counter(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def corpus_bleu4_pct(predictions: Sequence[Any], references: Sequence[Any]) -> float:
    """Compute a local, smoothed corpus BLEU-4 percentage.

    The implementation uses the BLEU brevity penalty and clipped n-gram
    precisions.  It applies effective order for short responses and smoothing
    only to zero higher-order precisions, avoiding an all-zero score for the
    small client test partitions common in federated instruction tuning.
    """

    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    if not predictions:
        return 0.0

    matches_by_order = [0] * 4
    possible_by_order = [0] * 4
    prediction_length = 0
    reference_length = 0

    for prediction, reference in zip(predictions, references):
        prediction_tokens = metric_tokenize(prediction)
        reference_tokens = metric_tokenize(reference)
        prediction_length += len(prediction_tokens)
        reference_length += len(reference_tokens)
        for order in range(1, 5):
            prediction_ngrams = _ngram_counter(prediction_tokens, order)
            reference_ngrams = _ngram_counter(reference_tokens, order)
            matches_by_order[order - 1] += sum(
                min(count, reference_ngrams[ngram])
                for ngram, count in prediction_ngrams.items()
            )
            possible_by_order[order - 1] += max(len(prediction_tokens) - order + 1, 0)

    if prediction_length == 0:
        return 0.0

    effective_orders = [order for order, possible in enumerate(possible_by_order, start=1) if possible]
    if not effective_orders:
        return 0.0

    log_precision_sum = 0.0
    for order in effective_orders:
        matches = matches_by_order[order - 1]
        possible = possible_by_order[order - 1]
        if matches:
            precision = matches / possible
        elif order == 1:
            return 0.0
        else:
            # Chen-Cherry-style low-order smoothing for a missing n-gram.
            precision = 1.0 / (2.0 * possible)
        log_precision_sum += math.log(precision)

    geo_mean = math.exp(log_precision_sum / len(effective_orders))
    brevity_penalty = (
        1.0
        if prediction_length >= reference_length
        else math.exp(1.0 - reference_length / prediction_length)
    )
    return 100.0 * brevity_penalty * geo_mean


def compute_generation_metrics(predictions: Sequence[Any], references: Sequence[Any]) -> dict[str, Any]:
    """Return ROUGE-L F1 and corpus BLEU-4 for aligned generations.

    ``rougeL_f1`` is an arithmetic mean of per-example F1 values in ``[0, 1]``.
    ``bleu4_pct`` is corpus BLEU-4 in percentage points, in ``[0, 100]``.
    """

    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    if not predictions:
        return {"examples": 0, "rougeL_f1": None, "bleu4_pct": None}
    return {
        "examples": len(predictions),
        "rougeL_f1": sum(rouge_l_f1(prediction, reference) for prediction, reference in zip(predictions, references))
        / len(predictions),
        "bleu4_pct": corpus_bleu4_pct(predictions, references),
    }


def build_instruction_prompt(instruction: Any, *, template_name: str = "alpaca") -> str:
    """Render an inference prompt that ends immediately after the response tag."""

    try:
        template, _ = TEMPLATE_DICT[template_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown template `{template_name}`; expected one of {sorted(TEMPLATE_DICT)}."
        ) from error
    return template.format(str(instruction or ""), "", "")


def _dataset_records(dataset: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    records = list(dataset)
    for index, record in enumerate(records):
        missing = {"instruction", "response"}.difference(record)
        if missing:
            raise ValueError(f"Held-out example {index} is missing required columns: {sorted(missing)}")
    return records


def _infer_task_name(records: Sequence[Mapping[str, Any]]) -> str:
    categories = []
    for record in records:
        category = str(record.get("category", "")).strip()
        if category and category not in categories:
            categories.append(category)
    return ", ".join(categories) if categories else "iid"


def evaluate_instruction_predictions(
    heldout_dataset: Iterable[Mapping[str, Any]],
    predictions: Sequence[Any],
    *,
    include_predictions: bool = False,
) -> dict[str, Any]:
    """Score already-generated answers against a single client's held-out set."""

    records = _dataset_records(heldout_dataset)
    if len(predictions) != len(records):
        raise ValueError(
            f"Expected {len(records)} predictions for the held-out dataset, got {len(predictions)}."
        )
    references = [str(record["response"] or "") for record in records]
    result = compute_generation_metrics(predictions, references)
    predicted_tokens = [metric_tokenize(p) for p in predictions]
    reference_tokens = [metric_tokenize(r) for r in references]
    pred_len = sum(map(len, predicted_tokens))
    ref_len = sum(map(len, reference_tokens))
    result["bleu_brevity_penalty"] = min(1.0, math.exp(min(0.0, 1.0 - ref_len / pred_len))) if pred_len else 0.0
    result["bleu_prediction_tokens"] = pred_len
    result["bleu_reference_tokens"] = ref_len
    for order in range(1, 5):
        matches = sum(sum((_ngram_counter(p, order) & _ngram_counter(r, order)).values())
                      for p, r in zip(predicted_tokens, reference_tokens))
        possible = sum(max(len(p) - order + 1, 0) for p in predicted_tokens)
        result[f"bleu_precision_{order}_raw"] = matches / possible if possible else 0.0
    if include_predictions:
        result["predictions"] = [str(prediction) for prediction in predictions]
        result["references"] = references
    return result


def _resolve_input_device(model: Any, requested_device: Optional[str | torch.device]) -> torch.device:
    if requested_device is not None:
        return torch.device(requested_device)
    try:
        embedding_weight = model.get_input_embeddings().weight
        if embedding_weight.device.type != "meta":
            return embedding_weight.device
    except (AttributeError, StopIteration):
        pass
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def generate_instruction_responses(
    model: Any,
    tokenizer: Any,
    heldout_dataset: Iterable[Mapping[str, Any]],
    *,
    template_name: str = "alpaca",
    batch_size: int = 1,
    max_new_tokens: int = 128,
    max_prompt_length: Optional[int] = None,
    generation_kwargs: Optional[Mapping[str, Any]] = None,
    device: Optional[str | torch.device] = None,
    generation_records: Optional[list] = None,
) -> list[str]:
    """Greedily generate one completion per held-out instruction.

    The caller is responsible for loading the desired global or personalized
    adapter state before invoking this function.  ``heldout_dataset`` is never
    mixed with a training partition.
    """

    if batch_size < 1:
        raise ValueError("`batch_size` must be at least one.")
    if max_new_tokens < 1:
        raise ValueError("`max_new_tokens` must be at least one.")
    records = _dataset_records(heldout_dataset)
    if not records:
        return []

    prompts = [build_instruction_prompt(record["instruction"], template_name=template_name) for record in records]
    input_device = _resolve_input_device(model, device)
    generation_options = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        # Training disables cache to support gradient checkpointing; inference
        # must explicitly re-enable it or long held-out generations become
        # needlessly quadratic on decoder-only models.
        "use_cache": True,
        "return_dict_in_generate": False,
    }
    if generation_kwargs:
        generation_options.update(generation_kwargs)
    if generation_options.get("pad_token_id") is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        generation_options["pad_token_id"] = (
            pad_token_id if pad_token_id is not None else getattr(tokenizer, "eos_token_id", None)
        )
    if generation_options.get("eos_token_id") is None:
        generation_options["eos_token_id"] = getattr(tokenizer, "eos_token_id", None)

    was_training = bool(getattr(model, "training", False))
    original_padding_side = getattr(tokenizer, "padding_side", None)
    original_truncation_side = getattr(tokenizer, "truncation_side", None)
    predictions: list[str] = []
    try:
        model.eval()
        # Decoder-only generation needs left padding so every continuation
        # begins after the final real prompt token in a batch.
        if original_padding_side is not None:
            tokenizer.padding_side = "left"
        # The response marker is at the prompt's end.  Keeping the right-hand
        # side during truncation prevents a long instruction from erasing it.
        if max_prompt_length is not None and original_truncation_side is not None:
            tokenizer.truncation_side = "left"
        with torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                prompt_batch = prompts[start : start + batch_size]
                tokenized = tokenizer(
                    prompt_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=max_prompt_length is not None,
                    max_length=max_prompt_length,
                )
                tokenized = _move_batch_to_device(tokenized, input_device)
                generated = model.generate(**tokenized, **generation_options)
                sequences = getattr(generated, "sequences", generated)
                if bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False)):
                    completion_ids = sequences
                else:
                    completion_ids = sequences[:, tokenized["input_ids"].shape[1] :]
                if generation_records is not None:
                    eos = generation_options.get("eos_token_id")
                    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
                    for row in completion_ids.tolist():
                        end = next((i for i, token in enumerate(row) if token in eos_ids), None)
                        count = end + 1 if end is not None else len(row)
                        generation_records.append({
                            "generated_tokens": count,
                            "termination": "eos" if end is not None else (
                                "length_limit" if count >= generation_options["max_new_tokens"] else "other"),
                        })
                predictions.extend(tokenizer.batch_decode(completion_ids, skip_special_tokens=True))
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
        if original_truncation_side is not None:
            tokenizer.truncation_side = original_truncation_side
        if was_training:
            model.train()
    return [prediction.strip() for prediction in predictions]


def evaluate_instruction_generations(
    model: Any,
    tokenizer: Any,
    heldout_dataset: Iterable[Mapping[str, Any]],
    **generation_options: Any,
) -> dict[str, Any]:
    """Generate and score one model on one client's held-out task partition."""

    records = _dataset_records(heldout_dataset)
    include_predictions = bool(generation_options.pop("include_predictions", False))
    generation_records = []
    predictions = generate_instruction_responses(model, tokenizer, records,
        generation_records=generation_records, **generation_options)
    result = evaluate_instruction_predictions(
        records,
        predictions,
        include_predictions=include_predictions,
    )
    if records:
        repeats = []
        for prediction in predictions:
            grams = _ngram_counter(metric_tokenize(prediction), 4)
            total = sum(grams.values())
            repeats.append(1.0 - len(grams) / total if total else 0.0)
        result.update({
            "eos_rate": sum(r["termination"] == "eos" for r in generation_records) / len(records),
            "length_limit_rate": sum(r["termination"] == "length_limit" for r in generation_records) / len(records),
            "mean_generated_tokens": sum(r["generated_tokens"] for r in generation_records) / len(records),
            "repeated_4gram_fraction": sum(repeats) / len(records),
        })
        if include_predictions:
            result["generation_records"] = generation_records
    return result


def _aggregate_client_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Optional[dict[str, float]]]:
    nonempty = [record for record in records if record.get("examples", 0)]
    if not nonempty:
        return {"macro_average": None, "example_weighted_average": None}

    macro_average = {
        metric: sum(float(record[metric]) for record in nonempty) / len(nonempty)
        for metric in ("rougeL_f1", "bleu4_pct")
    }
    total_examples = sum(int(record["examples"]) for record in nonempty)
    weighted_average = {
        metric: sum(int(record["examples"]) * float(record[metric]) for record in nonempty)
        / total_examples
        for metric in ("rougeL_f1", "bleu4_pct")
    }
    return {"macro_average": macro_average, "example_weighted_average": weighted_average}


def evaluate_client_heldout_predictions(
    client_heldout_datasets: Sequence[Iterable[Mapping[str, Any]]],
    client_predictions: Sequence[Sequence[Any]],
    *,
    client_task_names: Optional[Sequence[str]] = None,
    include_predictions: bool = False,
) -> dict[str, Any]:
    """Score predictions on each client's own held-out task and aggregate them."""

    if len(client_heldout_datasets) != len(client_predictions):
        raise ValueError("A prediction sequence is required for every client held-out dataset.")
    if client_task_names is not None and len(client_task_names) != len(client_heldout_datasets):
        raise ValueError("`client_task_names` must have one entry per client.")

    per_client = []
    for client, (dataset, predictions) in enumerate(zip(client_heldout_datasets, client_predictions)):
        records = _dataset_records(dataset)
        result = evaluate_instruction_predictions(
            records, predictions, include_predictions=include_predictions
        )
        result.update(
            {
                "client": client,
                "task": client_task_names[client] if client_task_names else _infer_task_name(records),
            }
        )
        per_client.append(result)
    return {"per_client": per_client, **_aggregate_client_metrics(per_client)}


def evaluate_client_heldout_generations(
    model: Any,
    tokenizer: Any,
    client_heldout_datasets: Sequence[Iterable[Mapping[str, Any]]],
    *,
    prepare_client: Optional[Callable[[int], None]] = None,
    client_task_names: Optional[Sequence[str]] = None,
    include_predictions: bool = False,
    **generation_options: Any,
) -> dict[str, Any]:
    """Personalized held-out evaluation, keeping every client on its own task.

    ``prepare_client`` is invoked immediately before each client's generation.
    For FedSA it should load the shared server A plus that client's private B;
    for FedAvg/FRLoRA it can load the final global state or be omitted when the
    model is already in that state.
    """

    if client_task_names is not None and len(client_task_names) != len(client_heldout_datasets):
        raise ValueError("`client_task_names` must have one entry per client.")

    all_predictions = []
    for client, dataset in enumerate(client_heldout_datasets):
        if prepare_client is not None:
            prepare_client(client)
        all_predictions.append(
            generate_instruction_responses(model, tokenizer, dataset, **generation_options)
        )
    return evaluate_client_heldout_predictions(
        client_heldout_datasets,
        all_predictions,
        client_task_names=client_task_names,
        include_predictions=include_predictions,
    )
