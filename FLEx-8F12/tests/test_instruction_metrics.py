"""Offline generation-metric tests for client-held-out instruction evaluation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
import torch

from evaluation.instruction_metrics import (
    build_instruction_prompt,
    compute_generation_metrics,
    evaluate_client_heldout_predictions,
    evaluate_instruction_generations,
)


class InstructionMetricTests(unittest.TestCase):
    def test_eos_metadata_ignores_post_eos_padding_and_detects_cap(self):
        class Tokenizer:
            pad_token_id = eos_token_id = 2
            padding_side = truncation_side = 'right'
            def __call__(self, prompts, **kwargs):
                return {'input_ids': torch.tensor([[1, 1]] * len(prompts)),
                        'attention_mask': torch.ones(len(prompts), 2, dtype=torch.long)}
            def batch_decode(self, rows, **kwargs):
                return ['a b', 'a b c d', 'a b c']
        class Model(torch.nn.Module):
            config = SimpleNamespace(is_encoder_decoder=False)
            def generate(self, **kwargs):
                return torch.tensor([[1, 1, 7, 2, 2, 2], [1, 1, 7, 8, 9, 10],
                                     [1, 1, 7, 8, 9, 2]])
        model, tokenizer = Model(), Tokenizer()
        result = evaluate_instruction_generations(model, tokenizer,
            [{'instruction': 'test', 'response': 'a b c d'}] * 3,
            batch_size=3, max_new_tokens=4, include_predictions=True)
        self.assertEqual(result['generation_records'], [
            {'generated_tokens': 2, 'termination': 'eos'},
            {'generated_tokens': 4, 'termination': 'length_limit'},
            {'generated_tokens': 4, 'termination': 'eos'}])
        self.assertAlmostEqual(result['eos_rate'], 2/3)
        self.assertAlmostEqual(result['length_limit_rate'], 1/3)
        self.assertEqual(tokenizer.padding_side, 'right')
        self.assertTrue(model.training)

    def test_perfect_predictions_score_one_rouge_and_100_bleu(self):
        predictions = ["The cat sat on the mat.", "A clear blue sky."]
        result = compute_generation_metrics(predictions, list(predictions))

        self.assertEqual(result["examples"], 2)
        self.assertAlmostEqual(result["rougeL_f1"], 1.0)
        self.assertAlmostEqual(result["bleu4_pct"], 100.0)

    def test_client_heldout_results_keep_task_boundaries_and_aggregate(self):
        client_heldout = [
            [
                {
                    "instruction": "answer task a",
                    "response": "alpha beta gamma delta",
                    "category": "task_a",
                }
            ],
            [
                {
                    "instruction": "answer task b",
                    "response": "one two three four",
                    "category": "task_b",
                },
                {
                    "instruction": "answer task b again",
                    "response": "five six seven eight",
                    "category": "task_b",
                },
            ],
        ]
        result = evaluate_client_heldout_predictions(
            client_heldout,
            [
                ["alpha beta gamma delta"],
                ["one two three four", "five six seven eight"],
            ],
        )

        self.assertEqual([record["task"] for record in result["per_client"]], ["task_a", "task_b"])
        self.assertEqual([record["examples"] for record in result["per_client"]], [1, 2])
        self.assertAlmostEqual(result["macro_average"]["rougeL_f1"], 1.0)
        self.assertAlmostEqual(result["example_weighted_average"]["bleu4_pct"], 100.0)

    def test_prompt_ends_at_the_selected_response_marker(self):
        prompt = build_instruction_prompt("Say hello", template_name="alpaca")
        self.assertTrue(prompt.endswith("### Response: "))
        self.assertNotIn("Say hello", prompt.split("### Response:", maxsplit=1)[1])


if __name__ == "__main__":
    unittest.main()
