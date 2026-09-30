"""In-memory Dolly/FLAN schema and task-partition tests."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from datasets import Dataset

from federated_learning.split_dataset import get_dataset_this_round, split_train_eval_datasets
from utils.process_dataset import process_sft_dataset


class TaskDataPipelineTests(unittest.TestCase):
    def test_dolly_and_flan_normalization_and_task_holdout(self):
        raw_dolly = Dataset.from_dict(
            {
                "instruction": ["question"] * 12,
                "context": ["context"] * 12,
                "response": [f"answer-{index}" for index in range(12)],
                "category": ["closed_qa"] * 3
                + ["information_extraction"] * 3
                + ["classification"] * 3
                + ["summarization"] * 3,
            }
        )
        dataset = process_sft_dataset("dolly", raw_dolly, dataset_sample=None)
        self.assertEqual(dataset.column_names, ["instruction", "response", "category"])
        fed_args = SimpleNamespace(
            num_clients=4,
            split_strategy="task",
            category_list=["closed_qa", "information_extraction", "classification", "summarization"],
            whether_same_sample=False,
            eval_fraction=1 / 3,
        )
        script_args = SimpleNamespace(
            seed=7,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=1,
            task_column=None,
        )
        train, heldout = split_train_eval_datasets(fed_args, script_args, dataset)
        self.assertEqual([len(partition) for partition in train], [2, 2, 2, 2])
        self.assertEqual([len(partition) for partition in heldout], [1, 1, 1, 1])
        for client_train, client_test in zip(train, heldout):
            self.assertTrue(
                set(client_train["response"]).isdisjoint(client_test["response"]),
                "A client's held-out records must never enter its local training partition.",
            )
        self.assertEqual(len(get_dataset_this_round(train[0], 0, fed_args, script_args, client_id=0)), 1)

        raw_flan = Dataset.from_dict(
            {"inputs": ["prompt"], "targets": [["first", "second"]], "task_name": ["toy_task"]}
        )
        flan = process_sft_dataset("flan", raw_flan, dataset_sample=None)
        self.assertEqual(flan[0]["response"], "first")
        self.assertEqual(flan[0]["category"], "toy_task")


if __name__ == "__main__":
    unittest.main()
