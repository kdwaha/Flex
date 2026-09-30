"""Deterministic dataset partitioning for federated instruction tuning."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Optional, Sequence, Tuple

from datasets import load_dataset

from utils import alpaca_format


_TASK_SPLIT_STRATEGIES = {"no-iid", "non-iid", "noniid", "task"}


def _as_positive_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`{name}` must be an integer, got {value!r}.") from error
    if result <= 0:
        raise ValueError(f"`{name}` must be positive, got {result}.")
    return result


def _seed_from_script_args(script_args, default: int = 2023) -> int:
    value = getattr(script_args, "seed", default)
    try:
        return int(default if value is None else value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`seed` must be an integer, got {value!r}.") from error


def _client_seed(base_seed: int, round_index: int, client_id: Any) -> int:
    """Create a stable seed without depending on Python's randomized hash()."""
    encoded_id = str(client_id).encode("utf-8")
    client_hash = int.from_bytes(hashlib.blake2b(encoded_id, digest_size=4).digest(), "big")
    return (base_seed + (int(round_index) * 1_000_003) + client_hash) % (2**32)


def _task_column(dataset, script_args) -> str:
    """Find the normalized task label, with a graceful raw-dataset fallback."""
    columns = list(dataset.column_names)
    # SFT preprocessing intentionally maps a user-provided task_column into
    # `category`, so prefer that unified name whenever it is available.
    if "category" in columns:
        return "category"

    requested_column = getattr(script_args, "task_column", None)
    if requested_column:
        if requested_column in columns:
            return requested_column
        raise ValueError(
            f"`task_column={requested_column!r}` was requested for task/no-iid "
            f"splitting, but dataset columns are {columns}."
        )

    for candidate in ("task_name", "task"):
        if candidate in columns:
            return candidate
    raise ValueError(
        "Task/no-iid splitting requires a `category` column (normally produced "
        "by process_sft_dataset) or an explicit valid `task_column`. "
        f"Dataset columns are {columns}."
    )


def _client_categories(fed_args, num_clients: int, label: str) -> Sequence[str]:
    categories = getattr(fed_args, "category_list", None)
    if categories is None:
        raise ValueError(
            f"`category_list` is required for {label} splitting and must contain "
            "one task label per client."
        )
    if isinstance(categories, str):
        raise ValueError(
            "`category_list` must be a list of task labels, not a single string. "
            "For one client, pass e.g. `--category_list my_task`."
        )
    if len(categories) < num_clients:
        raise ValueError(
            f"{label} splitting needs {num_clients} entries in `category_list`, "
            f"but received {len(categories)}: {list(categories)!r}."
        )

    selected = []
    for client_id, category in enumerate(categories[:num_clients]):
        if category is None or not str(category).strip():
            raise ValueError(
                f"`category_list[{client_id}]` must be a non-empty task label; "
                f"got {category!r}."
            )
        selected.append(str(category))
    return selected


def _available_labels(dataset, column: str, limit: int = 20):
    try:
        values = dataset.unique(column)
    except Exception:  # pragma: no cover - defensive for non-HF Dataset-like inputs
        return "<unavailable>"
    values = sorted({str(value) for value in values})
    summary = values[:limit]
    if len(values) > limit:
        summary.append(f"... (+{len(values) - limit} more)")
    return summary


def _filter_client_dataset(dataset, column: str, label: str, client_id: int, split_name: str):
    filtered_dataset = dataset.filter(
        lambda example, expected_label=label: str(example[column]) == expected_label,
        desc=f"Assigning {split_name} {label!r} to client {client_id}",
    )
    if len(filtered_dataset) == 0:
        available = _available_labels(dataset, column)
        raise ValueError(
            f"Client {client_id} was assigned {split_name} {label!r}, but that "
            f"label has no examples in column `{column}`. Available labels: {available}."
        )
    return filtered_dataset


def split_dataset(fed_args, script_args, dataset):
    """Split a dataset into client datasets.

    This public API deliberately continues to return *only a list* so existing
    SFT and DPO callers remain compatible.  Use ``split_train_eval_datasets``
    when local held-out evaluation sets are desired.
    """
    if dataset is None:
        raise ValueError("Cannot split a missing dataset.")
    if len(dataset) == 0:
        raise ValueError("Cannot split an empty dataset.")

    num_clients = _as_positive_int(getattr(fed_args, "num_clients", None), "num_clients")
    split_strategy = str(getattr(fed_args, "split_strategy", "iid") or "iid").strip().lower()
    dataset = dataset.shuffle(seed=_seed_from_script_args(script_args))

    if split_strategy == "iid":
        local_datasets = [dataset.shard(num_clients, client_id) for client_id in range(num_clients)]
        empty_clients = [client_id for client_id, local_dataset in enumerate(local_datasets) if len(local_dataset) == 0]
        if empty_clients:
            raise ValueError(
                f"IID splitting produced empty client datasets for clients {empty_clients}. "
                f"Dataset has {len(dataset)} examples but num_clients={num_clients}."
            )
        return local_datasets

    if split_strategy in _TASK_SPLIT_STRATEGIES:
        column = _task_column(dataset, script_args)
        categories = _client_categories(fed_args, num_clients, "task/no-iid")
        limit_to_same_sample = bool(getattr(fed_args, "whether_same_sample", False))
        same_sample_count = _as_positive_int(getattr(fed_args, "same_sample_count", 100), "same_sample_count")

        local_datasets = []
        for client_id, category in enumerate(categories):
            print(f"Client {client_id} - {category}")
            filtered_dataset = _filter_client_dataset(
                dataset, column, category, client_id, "task"
            )
            if limit_to_same_sample:
                filtered_dataset = filtered_dataset.select(
                    range(min(same_sample_count, len(filtered_dataset)))
                )
            local_datasets.append(filtered_dataset)
        return local_datasets

    if split_strategy == "dirichlet":
        if "client_id" not in dataset.column_names:
            raise ValueError(
                "Dirichlet splitting requires a `client_id` column. Use the "
                "`dolly/dirichlet` loader or provide pre-partitioned client data."
            )
        client_labels = _client_categories(fed_args, num_clients, "dirichlet")
        limit_to_same_sample = bool(getattr(fed_args, "whether_same_sample", False))
        same_sample_count = _as_positive_int(getattr(fed_args, "same_sample_count", 100), "same_sample_count")

        local_datasets = []
        for client_id, client_label in enumerate(client_labels):
            print(f"Client {client_id} - {client_label}")
            filtered_dataset = _filter_client_dataset(
                dataset, "client_id", client_label, client_id, "client_id"
            )
            if limit_to_same_sample:
                filtered_dataset = filtered_dataset.select(
                    range(min(same_sample_count, len(filtered_dataset)))
                )
            local_datasets.append(filtered_dataset)
        return local_datasets

    supported = ["iid", "task", "no-iid", "dirichlet"]
    raise ValueError(
        f"Unknown split_strategy={split_strategy!r}. Supported strategies: {supported}."
    )


def split_train_eval_datasets(
    fed_args,
    script_args,
    dataset,
    eval_fraction: Optional[float] = None,
) -> Tuple[list, list]:
    """Return deterministic per-client train and held-out evaluation datasets.

    The split always preserves at least one training item for a non-empty
    client.  A one-example client consequently gets an empty evaluation set;
    callers can decide whether to skip evaluation for that client.
    """
    local_datasets = split_dataset(fed_args, script_args, dataset)
    configured_fraction = getattr(fed_args, "eval_fraction", 0.0)
    fraction = configured_fraction if eval_fraction is None else eval_fraction
    try:
        fraction = float(0.0 if fraction is None else fraction)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`eval_fraction` must be a float in [0, 1), got {fraction!r}.") from error
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"`eval_fraction` must be in [0, 1), got {fraction}.")

    train_datasets, eval_datasets = [], []
    base_seed = _seed_from_script_args(script_args)
    for client_id, local_dataset in enumerate(local_datasets):
        if fraction == 0.0 or len(local_dataset) <= 1:
            train_datasets.append(local_dataset)
            eval_datasets.append(local_dataset.select([]))
            continue

        eval_count = max(1, int(round(len(local_dataset) * fraction)))
        eval_count = min(eval_count, len(local_dataset) - 1)
        split = local_dataset.train_test_split(
            test_size=eval_count,
            seed=_client_seed(base_seed, 0, client_id),
            shuffle=True,
        )
        train_datasets.append(split["train"])
        eval_datasets.append(split["test"])
    return train_datasets, eval_datasets


def get_dataset_this_round(dataset, round, fed_args, script_args, client_id: Optional[Any] = None):
    """Select one deterministic local training subset for a federated round.

    Supplying ``client_id`` gives each client its own reproducible stream.  The
    optional argument preserves the old DPO call signature and its historical
    ``Random(round)`` sampling behavior when omitted.
    """
    if dataset is None:
        raise ValueError("Cannot sample from a missing local dataset.")
    if len(dataset) == 0:
        raise ValueError("Cannot sample from an empty local dataset.")

    try:
        round_index = int(round)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`round` must be an integer, got {round!r}.") from error

    try:
        batch_size = int(getattr(script_args, "batch_size", 1))
        gradient_accumulation_steps = int(getattr(script_args, "gradient_accumulation_steps", 1))
        max_steps = int(getattr(script_args, "max_steps", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("batch_size, gradient_accumulation_steps, and max_steps must be integers.") from error
    if batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive.")

    # `max_steps=-1` conventionally means no step cap; use all available local
    # data in that case.  Zero steps result in a valid empty round dataset.
    if max_steps < 0:
        num_to_sample = len(dataset)
    else:
        num_to_sample = min(
            batch_size * gradient_accumulation_steps * max_steps,
            len(dataset),
        )
    if num_to_sample == 0:
        return dataset.select([])

    if client_id is None:
        # Exact legacy behavior, without mutating the process-wide random state.
        random_generator = random.Random(round_index)
    else:
        random_generator = random.Random(
            _client_seed(_seed_from_script_args(script_args), round_index, client_id)
        )
    random_indices = random_generator.sample(range(len(dataset)), num_to_sample)
    return dataset.select(random_indices)


def get_eval_dataset_this_round(category):
    print(category)
    if category[0] == 'closed_qa':
        dataset = load_dataset("json", data_files='/home/liuf/OpenFedLLM/test_datasets/test_closed_qa.json')['train']
    elif category[0] == 'information_extraction':
        dataset = load_dataset("json", data_files='/home/liuf/OpenFedLLM/test_datasets/test_information_extraction.json')['train']
    elif category[0] == 'classification':
        dataset = load_dataset("json", data_files='/home/liuf/OpenFedLLM/test_datasets/test_classification.json')['train']
    elif category[0] == 'summarization':
        dataset = load_dataset("json", data_files='/home/liuf/OpenFedLLM/test_datasets/test_summarization.json')['train']
    else:
        raise ValueError(f'No evaluation dataset configured for category {category!r}.')

    dataset = dataset.rename_column("context", "input")
    dataset = dataset.rename_column("response", "output")
    dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc="Preprocessing for unified format.")

    assert len(dataset) == 100, \
        f"Expect {100}, but {len(dataset)}"

    return dataset
