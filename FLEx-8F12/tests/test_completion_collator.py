"""Offline regressions for EOS supervision, prompt masking, and true padding."""
import unittest
import warnings

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from utils.completion_collator import EOSPreservingCompletionCollator


class CompletionCollatorTests(unittest.TestCase):
    def collator(self, padding_side="right", shared_pad=True):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(
                {"<unk>": 0, "<bos>": 1, "<eos>": 2, "response": 3,
                 "question": 4, "answer": 5, "longer": 6, "<pad>": 7}, unk_token="<unk>")),
            unk_token="<unk>", eos_token="<eos>",
            pad_token="<eos>" if shared_pad else "<pad>", padding_side=padding_side,
        )
        return EOSPreservingCompletionCollator([3], tokenizer=tokenizer)

    def test_real_eos_is_supervised_but_prompt_eos_and_padding_are_not(self):
        for side in ("left", "right"):
            for shared in (False, True):
                with self.subTest(side=side, shared=shared):
                    collator = self.collator(side, shared)
                    batch = collator([
                        {"input_ids": [1, 2, 3, 5, 2], "attention_mask": [1]*5},
                        {"input_ids": [1, 4, 3, 5, 6, 2], "attention_mask": [1]*6},
                    ])
                    self.assertEqual(int((batch["labels"] == 2).sum()), 2)
                    self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
                    for ids, mask, labels in zip(batch["input_ids"], batch["attention_mask"], batch["labels"]):
                        boundary = ids.tolist().index(3)
                        self.assertTrue(torch.all(labels[:boundary+1] == -100))
                        self.assertTrue(torch.equal(labels[boundary+1:][mask[boundary+1:].bool()],
                                                    ids[boundary+1:][mask[boundary+1:].bool()]))

    def test_truncation_does_not_invent_an_eos_label(self):
        batch = self.collator()([
            {"input_ids": [1, 3, 5], "attention_mask": [1]*3},
            {"input_ids": [1, 3, 5, 6, 2], "attention_mask": [1]*5},
        ])
        self.assertEqual(batch["labels"][0].tolist(), [-100, -100, 5, -100, -100])
        self.assertEqual(int((batch["labels"][1] == 2).sum()), 1)

    def test_missing_response_marker_keeps_entire_example_ignored(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            batch = self.collator()([{"input_ids": [1, 4, 2], "attention_mask": [1]*3}])
        self.assertTrue(torch.all(batch["labels"] == -100))

    def test_eos_contributes_to_cross_entropy_and_gradient(self):
        batch = self.collator()([{"input_ids": [1, 3, 5, 2], "attention_mask": [1]*4}])
        logits = torch.zeros(1, 4, 8, requires_grad=True)
        labels = batch["labels"][:, 1:]
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 8), labels.reshape(-1))
        loss.backward()
        self.assertLess(float(logits.grad[0, 2, 2]), 0.)  # increase EOS after the answer
        self.assertTrue(torch.all(logits.grad[0, 0] == 0))  # ignored prompt position


if __name__ == "__main__":
    unittest.main()
