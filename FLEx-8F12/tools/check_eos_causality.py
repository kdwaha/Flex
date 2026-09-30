"""One-round paired EOS-supervision diagnostic. Never saves model checkpoints."""
import copy
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from evaluation.federated_diagnostics import completion_batches, diagnostic_state, token_metrics
from evaluation.instruction_metrics import build_instruction_prompt, compute_generation_metrics, metric_tokenize, _ngram_counter
from federated_learning.fed_local_sft import SFTTrainerPFLAlign
from federated_learning.split_dataset import get_dataset_this_round, split_train_eval_datasets
from utils import get_dataset, process_sft_dataset, get_formatting_prompts_func
from utils.completion_collator import EOSPreservingCompletionCollator as PreserveResponseEOS


def main(output):
    # Import existing preprocessing helpers without passing this tool's CLI to config.py.
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        from main_sft import _response_template_token_ids, _keep_completion_examples
    finally:
        sys.argv = original_argv
    assert torch.cuda.is_available(), 'GPU 0 is required; no silent CPU training'
    output.mkdir(parents=True, exist_ok=False)
    source = Path('outputs/llama1b_gpu0/round30_constant_final_hessian/pflalign')
    config = json.loads(sorted(source.glob('*/args.json'))[-1].read_text())
    args, fed = SimpleNamespace(**config['script_args']), SimpleNamespace(**config['fed_args'])
    config['diagnostic'] = dict(rounds=1, local_epochs=5, arms=['original', 'preserve_eos'],
        algorithms=['fedavg', 'pflalign'], checkpoints=False, generation_max_new_tokens=500,
        trainer_seed=42, purpose='EOS supervision intervention; not long-run component ablation')
    (output/'config.json').write_text(json.dumps(config, indent=2))
    set_seed(2025)
    data = process_sft_dataset(args.dataset_name, get_dataset(args.dataset_name, args.local_data_dir),
                               args.dataset_sample, task_column=args.task_column)
    train, test = split_train_eval_datasets(fed, args, data)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32,
        device_map={'': 0}, attn_implementation='eager', local_files_only=True)
    model.generation_config = GenerationConfig.from_pretrained(args.model_name_or_path, local_files_only=True)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, bias='none',
        task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, local_files_only=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    formatter, marker = get_formatting_prompts_func('alpaca', tokenizer.eos_token)
    marker_ids = _response_template_token_ids(tokenizer, formatter, marker)
    train = [_keep_completion_examples(ds, tokenizer, formatter, marker_ids, 512, 'train') for ds in train]
    test = [_keep_completion_examples(ds, tokenizer, formatter, marker_ids, 512, 'test') for ds in test]
    assert [len(ds) for ds in train] == [15, 15, 16, 16]
    assert [len(ds) for ds in test] == [4, 3, 4, 3]
    original = DataCollatorForCompletionOnlyLM(marker_ids, tokenizer=tokenizer)
    fixed = PreserveResponseEOS(marker_ids, tokenizer=tokenizer)
    records = []

    audit = []
    for client, ds in enumerate(train):
        old = completion_batches(tokenizer, formatter, original, ds, 512)
        new = completion_batches(tokenizer, formatter, fixed, ds, 512)
        restored = 0
        for a, b in zip(old, new):
            change = a['labels'] != b['labels']
            assert torch.equal(a['input_ids'], b['input_ids'])
            assert torch.all(b['input_ids'][change] == tokenizer.eos_token_id)
            assert torch.all(b['attention_mask'][change] == 1)
            assert torch.all(a['labels'][change] == -100)
            restored += int(change.sum())
        audit.append(dict(client=client, examples=len(ds), restored_eos_targets=restored))
        assert restored > 0
    # Explicit padded-batch regression: real EOS restored, EOS-valued padding ignored.
    probe = tokenizer(formatter({'instruction':['say yes', 'say more'],
                                 'response':['yes', 'this is longer than yes']}))
    batch = fixed([dict(input_ids=ids, attention_mask=mask)
                   for ids, mask in zip(probe['input_ids'], probe['attention_mask'])])
    assert torch.all(batch['labels'][batch['attention_mask'] == 0] == -100)
    assert int((batch['labels'] == tokenizer.eos_token_id).sum()) == 2
    (output/'label_audit.json').write_text(json.dumps(audit, indent=2))

    def evaluate(algorithm, arm, client, dataset):
        examples = list(dataset)
        with diagnostic_state(model):
            old_batches = completion_batches(tokenizer, formatter, original, dataset, 512)
            new_batches = completion_batches(tokenizer, formatter, fixed, dataset, 512)
            scores = token_metrics(model, old_batches)
            eos_scores = token_metrics(model, new_batches)
            probabilities = []
            with torch.no_grad():
                for b in new_batches:
                    inp = {k:v.to('cuda:0') for k,v in b.items()}
                    positions = (inp['labels'][0] == tokenizer.eos_token_id).nonzero().flatten()
                    if len(positions):
                        logits = model(**inp, use_cache=False).logits[0, positions - 1].float()
                        probabilities.extend(logits.softmax(-1)[:, tokenizer.eos_token_id].tolist())
            predictions, generation = [], []
            tokenizer.padding_side, tokenizer.truncation_side = 'left', 'left'
            try:
                with torch.inference_mode():
                    for start in range(0, len(examples), 4):
                        rows = examples[start:start+4]
                        inputs = tokenizer([build_instruction_prompt(r['instruction']) for r in rows],
                            return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda:0')
                        sequences = model.generate(**inputs, do_sample=False, num_beams=1, use_cache=True,
                            max_new_tokens=500, pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id)
                        for ids in sequences[:, inputs['input_ids'].shape[1]:].tolist():
                            has_eos = tokenizer.eos_token_id in ids
                            n = ids.index(tokenizer.eos_token_id) + 1 if has_eos else len(ids)
                            ids = ids[:n]
                            pred = tokenizer.decode(ids, skip_special_tokens=True).strip()
                            tokens = metric_tokenize(pred)
                            ngrams = _ngram_counter(tokens, 4)
                            repetition = 1 - len(ngrams)/sum(ngrams.values()) if ngrams else 0.
                            predictions.append(pred)
                            generation.append(dict(token_ids=ids, new_tokens=n,
                                stop_reason='eos' if has_eos else 'max_new_tokens' if n==500 else 'other',
                                repeated_4gram_fraction=repetition))
            finally:
                tokenizer.padding_side, tokenizer.truncation_side = 'right', 'right'
        references = [r['response'] for r in examples]
        row = dict(algorithm=algorithm, arm=arm, client=client,
            task=examples[0]['category'] if client >= 0 else 'all_test',
            **scores, eos_inclusive_loss=eos_scores['test_loss'],
            reference_end_eos_probability=sum(probabilities)/len(probabilities) if probabilities else None,
            scored_eos_examples=len(probabilities), **compute_generation_metrics(predictions, references),
            eos_stop_count=sum(x['stop_reason']=='eos' for x in generation),
            max_length_count=sum(x['stop_reason']=='max_new_tokens' for x in generation),
            mean_repeat4=sum(x['repeated_4gram_fraction'] for x in generation)/len(generation),
            predictions=predictions, references=references, generation=generation)
        records.append(row)
        (output/'evaluations.json').write_text(json.dumps(records, indent=2))

    initial = copy.deepcopy(get_peft_model_state_dict(model))
    all_test = [r for ds in test for r in ds]
    evaluate('untrained', 'none', -1, all_test)
    for algorithm in ['fedavg', 'pflalign']:
        for arm, collator in [('original', original), ('preserve_eos', fixed)]:
            local_states = []
            losses = []
            for client in range(4):
                set_peft_model_state_dict(model, initial)
                dataset = get_dataset_this_round(train[client], 0, fed, args, client_id=client)
                cfg = SFTConfig(output_dir=str(output/'unused_trainer_output'),
                    per_device_train_batch_size=4, gradient_accumulation_steps=1,
                    learning_rate=2e-4, lr_scheduler_type='constant', num_train_epochs=5,
                    max_steps=-1, max_seq_length=512, report_to='none', save_strategy='no',
                    logging_strategy='no', eval_strategy='no', gradient_checkpointing=False,
                    seed=42, max_grad_norm=0.0 if algorithm=='pflalign' else 1.0,
                    disable_tqdm=True, optim='adamw_torch')
                assert cfg.save_strategy == 'no'
                kwargs = dict(model=model, args=cfg, train_dataset=dataset,
                              formatting_func=formatter, data_collator=collator)
                if algorithm == 'pflalign':
                    zero = {k:torch.zeros_like(v) for k,v in initial.items()}
                    trainer = SFTTrainerPFLAlign(**kwargs, beta=.9, epsilon=1e-12,
                        pflalign_state=dict(delta=zero, v=zero, P=zero,
                            local_steps=5*math.ceil(len(dataset)/4)))
                else:
                    trainer = SFTTrainer(**kwargs)
                loss = trainer.train().training_loss
                losses.append(loss)
                local_states.append(copy.deepcopy(get_peft_model_state_dict(model)))
                del trainer
                evaluate(algorithm, arm, client, test[client])
            weights = [len(ds)/sum(map(len,train)) for ds in train]
            aggregate = {k:sum(s[k]*w for s,w in zip(local_states,weights)) for k in initial}
            set_peft_model_state_dict(model, aggregate)
            evaluate(algorithm, arm, -1, all_test)
            (output/f'{algorithm}_{arm}_train.json').write_text(json.dumps(dict(losses=losses)))
            del local_states, aggregate
    (output/'complete.json').write_text(json.dumps(dict(complete=True, evaluation_records=len(records))))
    print(str(output), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
