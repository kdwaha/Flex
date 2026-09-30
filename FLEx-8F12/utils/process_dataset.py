"""Dataset loading and SFT-schema normalization helpers.

The training code expects ``instruction`` and ``response`` columns.  Federated
instruction tuning additionally needs a stable ``category`` column so that a
client can be assigned one or more tasks.  This module keeps the legacy dataset
names working while recognizing the schemas used by the public Dolly and FLAN
releases (including locally downloaded JSON/JSONL/Parquet files).
"""

from __future__ import annotations

import glob
import json
import os
from functools import partial
from typing import Any, Iterable, Optional, Sequence

import datasets
import pandas as pd
from datasets import concatenate_datasets, load_dataset, load_from_disk

from .conversation import get_conv_template


_DOLLY_ALIASES = {
    "dolly",
    "databricks/databricks-dolly-15k",
    "databricks-dolly-15k",
}
_FLAN_ALIASES = {"flan", "muennighoff/flan"}
_LOCAL_EXTENSIONS = {".json", ".jsonl", ".parquet"}


def _normalised_name(dataset_name: Optional[str]) -> str:
    return str(dataset_name or "").strip().lower().rstrip("/")


def _is_dolly_dataset(dataset_name: Optional[str]) -> bool:
    return _normalised_name(dataset_name) in _DOLLY_ALIASES


def _is_flan_dataset(dataset_name: Optional[str]) -> bool:
    return _normalised_name(dataset_name) in _FLAN_ALIASES


def _dataset_from_loaded_splits(loaded_dataset: Any, split: str):
    """Return a Dataset from either a Dataset or a DatasetDict."""
    if isinstance(loaded_dataset, datasets.Dataset):
        return loaded_dataset
    if split in loaded_dataset:
        return loaded_dataset[split]
    if "train" in loaded_dataset:
        return loaded_dataset["train"]
    available = list(loaded_dataset.keys())
    if not available:
        raise ValueError("The loaded dataset has no splits.")
    return loaded_dataset[available[0]]


def _load_local_dataset(local_data_dir: str, split: str = "train"):
    """Load a saved Dataset(/Dict) or a directory/file of JSON(L)/Parquet data."""
    local_path = os.path.abspath(os.path.expanduser(local_data_dir))
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local dataset path does not exist: {local_path}")

    # Datasets saved with Dataset.save_to_disk / DatasetDict.save_to_disk should
    # retain their original split metadata instead of being treated as raw JSON.
    if os.path.isdir(local_path) and (
        os.path.exists(os.path.join(local_path, "state.json"))
        or os.path.exists(os.path.join(local_path, "dataset_dict.json"))
    ):
        return _dataset_from_loaded_splits(load_from_disk(local_path), split)

    if os.path.isfile(local_path):
        data_files: Sequence[str] | str = local_path
    else:
        # A Hugging Face repository cloned locally commonly contains
        # train/validation/test subdirectories.  Respect the requested split
        # instead of accidentally merging every split into the training data.
        raw_data_root = os.path.join(local_path, split)
        if not os.path.isdir(raw_data_root):
            raw_data_root = local_path
        data_files = sorted(
            file_path
            for extension in _LOCAL_EXTENSIONS
            for file_path in glob.glob(
                os.path.join(raw_data_root, "**", f"*{extension}"), recursive=True
            )
        )
        if not data_files:
            raise FileNotFoundError(
                "No .json, .jsonl, or .parquet files were found under "
                f"{local_path}."
            )

    suffixes = {os.path.splitext(file_path)[1].lower() for file_path in ([data_files] if isinstance(data_files, str) else data_files)}
    if suffixes == {".parquet"}:
        builder = "parquet"
    elif suffixes.issubset({".json", ".jsonl"}):
        builder = "json"
    else:
        raise ValueError(
            "A local dataset must contain only JSON/JSONL files or only Parquet "
            f"files; found extensions: {sorted(suffixes)}."
        )
    return _dataset_from_loaded_splits(load_dataset(builder, data_files=data_files), split)


def _load_remote_dataset(dataset_name: str, split: str, dataset_config_name: Optional[str] = None):
    kwargs = {"split": split}
    if dataset_config_name:
        kwargs["name"] = dataset_config_name
    return load_dataset(dataset_name, **kwargs)


def get_dataset(
    dataset_name,
    local_data_dir=None,
    dataset_config_name: Optional[str] = None,
    split: str = "train",
):
    """Load a legacy dataset or an official/local Dolly or FLAN dataset.

    ``local_data_dir`` may point at a single JSON/JSONL/Parquet file, a folder
    containing such files, or a dataset saved with ``save_to_disk``.  Passing no
    local path for ``dolly`` and ``flan`` selects their public Hugging Face
    releases.  The optional configuration name is intentionally optional so
    existing two-argument callers continue to work.
    """
    dataset_name = str(dataset_name)

    if dataset_name == "dolly/dirichlet":
        if local_data_dir is None:
            raise ValueError("`dolly/dirichlet` requires `local_data_dir` with client_* folders.")
        return load_dirichlet_dataset(local_data_dir)

    # For the two instruction-tuning datasets, an explicit local data path is
    # authoritative.  This makes an offline experiment use exactly the data the
    # user downloaded rather than silently falling back to the Hub.
    if (_is_dolly_dataset(dataset_name) or _is_flan_dataset(dataset_name)) and local_data_dir:
        return _load_local_dataset(local_data_dir, split=split)

    if _is_dolly_dataset(dataset_name):
        return _load_remote_dataset("databricks/databricks-dolly-15k", split=split)
    if _is_flan_dataset(dataset_name):
        return _load_remote_dataset(
            "Muennighoff/flan",
            split=split,
            dataset_config_name=dataset_config_name,
        )

    if dataset_name == "gsm8k":
        source_name = f"{local_data_dir}{dataset_name}" if local_data_dir is not None else dataset_name
        return load_dataset(source_name, split=split, name="main")
    if dataset_name == "lighteval/MATH":
        source_name = f"{local_data_dir}{dataset_name}" if local_data_dir is not None else dataset_name
        return load_dataset(source_name, split=split, name="all")
    if dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        source_name = f"{local_data_dir}{dataset_name}" if local_data_dir is not None else dataset_name
        return load_dataset(source_name, split="train_sft")
    if dataset_name in {"codemathgen", "alpaca"} and local_data_dir:
        return _load_local_dataset(local_data_dir, split=split)

    # Preserve the previous prefix-based convention for all other datasets.
    source_name = f"{local_data_dir}{dataset_name}" if local_data_dir is not None else dataset_name
    return _load_remote_dataset(source_name, split=split, dataset_config_name=dataset_config_name)


def _as_text(value: Any) -> str:
    """Convert common dataset values to a deterministic, non-null string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        # FLAN targets can be a list of acceptable completions.  Training on the
        # first non-empty completion avoids turning a Python list into a prompt.
        for item in value:
            text = _as_text(item)
            if text.strip():
                return text
        return ""
    if isinstance(value, dict):
        for key in ("text", "value", "content", "name"):
            if key in value:
                text = _as_text(value[key])
                if text.strip():
                    return text
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _find_column(
    columns: Iterable[str],
    candidates: Sequence[str],
    *,
    required: bool = False,
    label: str = "column",
) -> Optional[str]:
    columns = list(columns)
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    if required:
        raise ValueError(
            f"Could not identify a {label}. Expected one of {list(candidates)}, "
            f"but dataset columns are {columns}."
        )
    return None


def _join_instruction_and_context(instruction: str, context: str) -> str:
    instruction = instruction.strip()
    context = context.strip()
    if not instruction:
        return context
    if not context:
        return instruction
    return f"{instruction}\n\n{context}"


def _normalise_instruction_dataset(
    dataset,
    dataset_name: str,
    *,
    instruction_column: Optional[str],
    context_column: Optional[str],
    response_column: str,
    category_column: Optional[str],
):
    """Map a Dolly/Alpaca/FLAN-like dataset onto the common SFT columns."""
    if len(dataset) == 0:
        raise ValueError(f"Cannot preprocess empty dataset `{dataset_name}`.")

    default_category = _normalised_name(dataset_name) or "default"

    def normalise_example(example):
        instruction = _as_text(example.get(instruction_column)) if instruction_column else ""
        context = _as_text(example.get(context_column)) if context_column else ""
        category = _as_text(example.get(category_column)) if category_column else default_category
        return {
            "instruction": _join_instruction_and_context(instruction, context),
            "response": _as_text(example.get(response_column)),
            "category": category.strip() or default_category,
        }

    # Keep client_id for the legacy dirichlet loader.  It is needed by
    # split_dataset(..., split_strategy="dirichlet") but is not part of the
    # regular SFT schema.
    columns_to_keep = {"instruction", "response", "category", "client_id"}
    removable_columns = [
        column for column in dataset.column_names if column not in columns_to_keep
    ]
    dataset = dataset.map(
        normalise_example,
        remove_columns=removable_columns,
        desc=f"Preprocessing {dataset_name} for unified format.",
    )
    ordered_columns = ["instruction", "response", "category"]
    if "client_id" in dataset.column_names:
        ordered_columns.append("client_id")
    return dataset.select_columns(ordered_columns)


def _normalise_known_sft_dataset(dataset_name: str, dataset, task_column: Optional[str]):
    """Normalize a supported SFT dataset by schema instead of brittle renames."""
    columns = dataset.column_names
    lower_name = _normalised_name(dataset_name)

    if lower_name == "tiger-lab/mathinstruct":
        df = pd.DataFrame(dataset).drop_duplicates(subset=["instruction"])
        dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
        columns = dataset.column_names

    # Retain support for the project\'s non-instruction-shaped benchmark data.
    if lower_name == "lighteval/math":
        return _normalise_instruction_dataset(
            dataset,
            dataset_name,
            instruction_column=_find_column(columns, ["problem"], required=True, label="instruction column"),
            context_column=None,
            response_column=_find_column(columns, ["solution"], required=True, label="response column"),
            category_column=task_column if task_column in columns else None,
        )
    if lower_name == "gsm8k":
        return _normalise_instruction_dataset(
            dataset,
            dataset_name,
            instruction_column=_find_column(columns, ["question"], required=True, label="instruction column"),
            context_column=None,
            response_column=_find_column(columns, ["answer"], required=True, label="response column"),
            category_column=task_column if task_column in columns else None,
        )

    # Preserve the old Medical Meadow behavior, where the explicit `input`
    # field is the user prompt and the dataset's `instruction` field is not.
    if lower_name == "medalpaca/medical_meadow_medical_flashcards":
        return _normalise_instruction_dataset(
            dataset,
            dataset_name,
            instruction_column=None,
            context_column=_find_column(columns, ["input"], required=True, label="instruction column"),
            response_column=_find_column(columns, ["output"], required=True, label="response column"),
            category_column=task_column if task_column in columns else None,
        )

    # Public FLAN releases use `inputs`, `targets`, and one of `task` /
    # `task_name`.  A user-selected task column takes precedence when supplied.
    flan_inputs_column = _find_column(columns, ["inputs"])
    flan_targets_column = _find_column(columns, ["targets"])
    if flan_inputs_column and flan_targets_column:
        category_candidates = ([task_column] if task_column else []) + ["task", "task_name", "category", "type"]
        return _normalise_instruction_dataset(
            dataset,
            dataset_name,
            instruction_column=flan_inputs_column,
            context_column=None,
            response_column=flan_targets_column,
            category_column=_find_column(columns, category_candidates),
        )

    # Dolly JSON is published as instruction/context/response/category.  Local
    # variants commonly use the Alpaca spelling instruction/input/output/type.
    response_column = _find_column(columns, ["response", "output"])
    instruction_column = _find_column(columns, ["instruction"])
    context_column = _find_column(columns, ["context", "input"])
    if response_column and (instruction_column or context_column):
        category_candidates = ([task_column] if task_column else []) + [
            "category",
            "type",
            "task_name",
            "task",
            "source",
        ]
        return _normalise_instruction_dataset(
            dataset,
            dataset_name,
            instruction_column=instruction_column,
            context_column=context_column,
            response_column=response_column,
            category_column=_find_column(columns, category_candidates),
        )

    available = ", ".join(columns)
    raise NotImplementedError(
        f"Dataset `{dataset_name}` is not supported. Expected a Dolly-like "
        "(instruction/context/response/category or instruction/input/output/type) "
        f"or FLAN-like (inputs/targets/task/task_name) schema; got [{available}]."
    )


def process_sft_dataset(dataset_name, dataset, dataset_sample, task_column: Optional[str] = None):
    """Convert supported SFT datasets to instruction/response/category columns.

    ``task_column`` can point at a dataset-specific task label.  It is optional
    to preserve callers written before task-aware FLAN splitting was introduced.
    """
    dataset = _normalise_known_sft_dataset(dataset_name, dataset, task_column)
    dataset = dataset.shuffle(seed=2025)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    return dataset

# def alpaca_format(example):
#     if example['input'] == "":
#         example["instruction"] = example["instruction"]
#     else:
#         example["instruction"] = example["instruction"] + " " + example['input']
#     example["response"] = example['output']
#     return example
def alpaca_format(example):
    if 'input' not in example :
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    if 'output' in example :
        example["response"] = example['output']
    return example

def process_dpo_dataset(dataset_name, dataset, template_name, dataset_sample):
    if dataset_name in ["Anthropic/hh-rlhf"]:
        dataset = dataset.map(partial(split_hh, template_name=template_name), load_from_cache_file=False)
    elif dataset_name in ["HuggingFaceH4/ultrafeedback_binarized"]:
        dataset = dataset.map(partial(split_ultrafeedback, template_name=template_name), load_from_cache_file=False)
        dataset = dataset.remove_columns(['prompt_id', 'messages', 'score_chosen', 'score_rejected'])
    
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    print(f">> ===== Data Example =====")
    print(dataset[0])
    print(f">> {'='*50}")
    return dataset
    
def find_common_prefix(str1, str2):
    prefix = ""
    for i in range(min(len(str1), len(str2))):
        if str1[i] == str2[i]:
            prefix += str1[i]
        else:
            break
    return prefix

def split_ultrafeedback(example, template_name="vicuna_v1.1"):
    conv_template = get_conv_template(template_name)

    conv_template.append_message(conv_template.roles[0], example["prompt"])
    conv_template.append_message(conv_template.roles[1], None)
    example["prompt"] = conv_template.get_prompt()
    example["chosen"] = " " + example["chosen"][1]["content"]       # There might need a space in the front.
    example["rejected"] = " " + example["rejected"][1]["content"]
    return example

def split_hh(example, template_name="vicuna_v1.1"):
    common_prefix = find_common_prefix(example["chosen"], example["rejected"])

    conv_template = get_conv_template(template_name)

    sentence = common_prefix
    human_prefix_len = len("\n\nHuman: ")
    assistant_prefix_len = len("\n\nAssistant: ")
    sentence = sentence[human_prefix_len:]
    turn = "user"
    while True:
        if turn == "user":
            index = sentence.find("\n\nAssistant: ")
            if index == -1:
                break
            else:
                conv_template.append_message(conv_template.roles[0], sentence[:index])
                turn = "assistant"
                sentence = sentence[index + assistant_prefix_len :]
        elif turn == "assistant":
            index = sentence.find("\n\nHuman: ")
            if index == -1:
                break
            else:
                conv_template.append_message(conv_template.roles[1], sentence[:index])
                turn = "user"
                sentence = sentence[index + human_prefix_len :]
    conv_template.append_message(conv_template.roles[1], None)
    example["prompt"] = conv_template.get_prompt()
    example["chosen"] = example["chosen"][len(common_prefix) - 1 :]     # -1 to include the space in the front.
    example["rejected"] = example["rejected"][len(common_prefix) - 1 :]
    return example



def count_subdirectories(root_dir: str) -> int:
    count = 0
    if os.path.exists(root_dir) and os.path.isdir(root_dir):
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                count += 1
    return count

def load_dirichlet_dataset(root_dir):
    all_datasets = []
    num_clients = count_subdirectories(root_dir)
    
    for i in range(1, num_clients + 1):
        client_id = f"client_{i}"
        train_file_path = os.path.join(root_dir, client_id, "train", "train_data.json")

        if os.path.exists(train_file_path):
            try:
                client_dataset = load_dataset("json", data_files=train_file_path, split="train")
                # 添加 'client_id' 列
                client_dataset = client_dataset.add_column("client_id", [client_id] * len(client_dataset))
                all_datasets.append(client_dataset)
                print(f"Successfully Load {train_file_path} ({len(client_dataset)} data).")
            except Exception as e:
                print(f"Load {train_file_path} error: {e}")
        else:
            print(f"Files are not exist: {train_file_path}")

    if all_datasets:
        merged_dataset = concatenate_datasets(all_datasets)
        print(f"\nConcat {len(all_datasets)} client datasets, total {len(merged_dataset)} data")
        return merged_dataset
    else:
        print("\nNO CLIENTS DATASETS!")
        return None
    
