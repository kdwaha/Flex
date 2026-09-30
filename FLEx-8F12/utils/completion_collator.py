"""Completion-only labels that retain real response EOS when PAD equals EOS."""

from trl import DataCollatorForCompletionOnlyLM


class EOSPreservingCompletionCollator(DataCollatorForCompletionOnlyLM):
    """Single-response instruction batches; never label padding or prompt EOS."""

    loss_spec = "response_tokens_including_real_eos_v1"

    def torch_call(self, examples):
        if self.instruction_template is not None:
            raise ValueError("EOSPreservingCompletionCollator supports single-response templates only.")
        batch = super().torch_call(examples)
        for row in range(len(examples)):
            ids = batch["input_ids"][row].tolist()
            ends = [i + len(self.response_token_ids)
                    for i in range(len(ids) - len(self.response_token_ids) + 1)
                    if ids[i:i + len(self.response_token_ids)] == self.response_token_ids]
            if not ends:
                continue
            for i in range(ends[-1], len(ids)):
                if ids[i] == self.tokenizer.eos_token_id and batch["attention_mask"][row, i]:
                    batch["labels"][row, i] = ids[i]
        return batch
