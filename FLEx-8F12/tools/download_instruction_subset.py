#!/usr/bin/env python3
"""Download a deterministic task-balanced Dolly or FLAN JSONL subset.

Examples:
  python tools/download_instruction_subset.py --dataset dolly \
    --tasks closed_qa information_extraction classification summarization \
    --samples-per-task 600 --output data/dolly_tasks.jsonl

For FLAN, list source tasks, then pass four names:
  python tools/download_instruction_subset.py --dataset flan --list-tasks
  python tools/download_instruction_subset.py --dataset flan --tasks ...
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("dolly", "flan"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--samples-per-task", type=int, default=600)
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--scan-limit", type=int, default=20000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples_per_task <= 0:
        raise ValueError("--samples-per-task must be positive.")

    if args.dataset == "dolly":
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        task_column = "category"
        if args.list_tasks:
            print("\n".join(sorted(dataset.unique(task_column))))
            return
        if not args.tasks:
            raise ValueError("Pass --tasks for a downloadable Dolly subset.")
        rows = []
        counts = Counter()
        requested = set(args.tasks)
        for example in dataset:
            task = str(example[task_column])
            if task in requested and counts[task] < args.samples_per_task:
                rows.append(example)
                counts[task] += 1
            if all(counts[task] == args.samples_per_task for task in requested):
                break
    else:
        # The aggregate FLAN stream is ordered by source task, so scanning its
        # first rows cannot discover a useful multi-task client assignment.
        # Instead list/download the repository's individual task JSONLs.
        repo_id = "Muennighoff/flan"
        task_column = "task_name"
        train_files = list_repo_files(repo_id, repo_type="dataset")
        available_tasks = sorted(
            Path(filename).stem[: -len("_train")]
            for filename in train_files
            if filename.startswith("train/") and filename.endswith("_train.jsonl")
        )
        if args.list_tasks:
            print("\n".join(available_tasks))
            return
        if not args.tasks:
            raise ValueError("Pass --tasks for a downloadable FLAN subset.")
        unavailable = sorted(set(args.tasks) - set(available_tasks))
        if unavailable:
            raise ValueError(
                f"Unknown FLAN task(s): {unavailable}. Run with --list-tasks to inspect choices."
            )
        counts = Counter()
        rows = []
        for task in args.tasks:
            filename = f"train/{task}_train.jsonl"
            local_file = hf_hub_download(repo_id, filename=filename, repo_type="dataset")
            task_stream = load_dataset("json", data_files=local_file, split="train", streaming=True)
            for index, example in enumerate(task_stream):
                if index >= args.scan_limit:
                    break
                example[task_column] = task
                rows.append(example)
                counts[task] += 1
                if counts[task] >= args.samples_per_task:
                    break

    missing = [task for task in args.tasks if counts[task] < args.samples_per_task]
    if missing:
        raise ValueError(
            f"Could not collect {args.samples_per_task} examples for {missing}. "
            "Use --list-tasks (FLAN) or select available task names."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_json(str(args.output), orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(rows)} examples to {args.output}; task counts: {dict(counts)}")


if __name__ == "__main__":
    main()
