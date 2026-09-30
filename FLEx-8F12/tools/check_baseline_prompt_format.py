"""Read-only pretrained TinyLlama prompt-format diagnostic; no checkpoints."""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('USE_TF', '0')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import get_peft_model, LoraConfig
from evaluation.instruction_metrics import build_instruction_prompt, compute_generation_metrics, metric_tokenize, _ngram_counter
from utils.process_dataset import _join_instruction_and_context


def main(output):
    assert torch.cuda.is_available()
    assert not output.exists()
    set_seed(2025)
    model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={'':0},
        torch_dtype=torch.float32, attn_implementation='eager', local_files_only=True)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, bias='none',
        task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj']))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, local_files_only=True)
    tokenizer.padding_side, tokenizer.truncation_side = 'left', 'left'
    root = Path('outputs/llama1b_gpu0/round30_constant_final_hessian/untrained')
    baseline = next(r for r in json.loads(sorted(root.glob('*/diagnostics.json'))[-1].read_text())
                    if r['kind']=='evaluation' and r['client']==-1)
    raw = [json.loads(x) for x in Path('data/dolly_tasks_4x20.jsonl').read_text().splitlines()]
    rows = []
    for ref in baseline['references']:
        match = [r for r in raw if r['response']==ref]
        assert len(match)==1
        rows.append(match[0])
    system = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'
    result = []
    for mode in ['alpaca','native_chat']:
        prompts = []
        for row in rows:
            instruction = _join_instruction_and_context(row['instruction'],row['context'])
            prompts.append(build_instruction_prompt(instruction) if mode=='alpaca' else
                tokenizer.apply_chat_template([{'role':'system','content':system},
                    {'role':'user','content':instruction}], tokenize=False, add_generation_prompt=True))
        samples = []
        with torch.inference_mode():
            for start in range(0,len(prompts),4):
                # Native template already serializes special tokens; no added BOS.
                inp = tokenizer(prompts[start:start+4], return_tensors='pt', padding=True,
                    truncation=True, max_length=512, add_special_tokens=(mode=='alpaca')).to('cuda:0')
                gen = model.generate(**inp, max_new_tokens=500, do_sample=False, num_beams=1,
                    use_cache=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                for ids in gen[:,inp['input_ids'].shape[1]:].tolist():
                    eos = tokenizer.eos_token_id in ids
                    ids = ids[:ids.index(tokenizer.eos_token_id)+1] if eos else ids
                    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                    grams = _ngram_counter(metric_tokenize(text),4)
                    samples.append(dict(prediction=text, token_ids=ids, new_tokens=len(ids),
                        stop_reason='eos' if eos else 'max_new_tokens',
                        repeat4=1-len(grams)/sum(grams.values()) if grams else 0.))
        result.append(dict(mode=mode, samples=samples,
            eos_stop_count=sum(s['stop_reason']=='eos' for s in samples),
            max_length_count=sum(s['stop_reason']=='max_new_tokens' for s in samples),
            mean_repeat4=sum(s['repeat4'] for s in samples)/len(samples),
            **compute_generation_metrics([s['prediction'] for s in samples], baseline['references'])))
    output.write_text(json.dumps(result,indent=2))
    print(str(output))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
