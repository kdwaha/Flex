import json
import os
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from rouge_score import rouge_scorer
# from nltk.translate.bleu_score import sentence_bleu
import nltk
from utils.template import TEMPLATE_DICT
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
# import nltk
# nltk.download('punkt_tab')

def evaluate_predictions(predictions, references, output_file=None):
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}

    for pred, ref in zip(predictions, references):
 
        scores = rouge.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
        
        # ref_tokens = nltk.word_tokenize(ref)
        # pred_tokens = nltk.word_tokenize(pred)

    n = len(predictions)
    for key in rouge_scores:
        rouge_scores[key] /= n

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"ROUGE-1: {rouge_scores['rouge1']:.4f}\n")
            f.write(f"ROUGE-2: {rouge_scores['rouge2']:.4f}\n")
            f.write(f"ROUGE-L: {rouge_scores['rougeL']:.4f}\n")
            # f.write(f"BLEU: {avg_bleu:.4f}\n")

    return rouge_scores

def generate_alpaca_answer(model, tokenizer, test_data_path, result_path, cuda, args, lora_path=None, base_model_path=None) :
    template = TEMPLATE_DICT['alpaca'][0]
    print(f">> You are using template: {template}")
    
    eval_set = load_dataset("json", data_files=test_data_path)['train']
    device = cuda

    # if lora_path is not None:
        
    #     tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    #     model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16).to(device)
    #     model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16).to(device)
    
    result_list = []
    total_length = len(eval_set)
    for i, example in enumerate(eval_set):
        instruction = template.format(
            f"{example['instruction']}{'input: ' + example['input'] if example['input'] else ''}",
            # f"{example['instruction']}",
            "", ""
        )[:-1]
        input_ids = tokenizer.encode(instruction, return_tensors="pt").to(device)
        output_ids = model.generate(inputs=input_ids, do_sample=False, num_beams=1, max_new_tokens=256)
        output_ids = output_ids[0][len(input_ids[0]):]
        result = tokenizer.decode(output_ids, skip_special_tokens=True)
        example['model_output'] = result
        example['generator'] = 'Qwen1.5-moe'
    
        # print(f"\nInput: \n{instruction}")
        # print(f"\nOutput: \n{result}")
        # print("="*100)
        result_list.append(example)
        with open(result_path, "w") as f:
            json.dump(result_list, f, indent=4)
        print(f"Finished [{i + 1}/{total_length}]")

    predictions = [item["model_output"] for item in result_list]
    references = [item["output"] for item in result_list] 

    rouge_output_file = os.path.splitext(result_path)[0] + "_rouge.txt"
    rouge_scores = evaluate_predictions(predictions, references, rouge_output_file)

    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

def generate_dolloy_answer(model, tokenizer, test_data_path, result_path, cuda, args, lora_path=None, base_model_path=None) :
    template = TEMPLATE_DICT['alpaca'][0]
    print(f">> You are using template: {template}")
    
    eval_set = load_dataset("json", data_files=test_data_path)['train']
    device = cuda

    # if lora_path is not None:
        
    #     tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    #     model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16).to(device)
    #     model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16).to(device)
    
    result_list = []
    total_length = len(eval_set)
    for i, example in enumerate(eval_set):
        instruction = template.format(
            f"{example['instruction']}{'context: ' + example['context'] if example['context'] else ''}",
            "", ""
        )[:-1]
        input_ids = tokenizer.encode(instruction, return_tensors="pt").to(device)
        output_ids = model.generate(inputs=input_ids, do_sample=False, num_beams=1, max_new_tokens=256)
        output_ids = output_ids[0][len(input_ids[0]):]
        result = tokenizer.decode(output_ids, skip_special_tokens=True)
        example['model_output'] = result
        example['generator'] = 'Qwen1.5-moe'
    
        # print(f"\nInput: \n{instruction}")
        # print(f"\nOutput: \n{result}")
        # print("="*100)
        result_list.append(example)
        with open(result_path, "w") as f:
            json.dump(result_list, f, indent=4)
        print(f"Finished [{i + 1}/{total_length}]")

    predictions = [item["model_output"] for item in result_list]
    references = [item["response"] for item in result_list]  

    rouge_output_file = os.path.splitext(result_path)[0] + "_rouge.txt"

    rouge_scores = evaluate_predictions(predictions, references, rouge_output_file)

    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        # print(f"BLEU: {avg_bleu:.4f}")

