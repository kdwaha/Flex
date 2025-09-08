import random
from datasets import load_dataset
from utils import alpaca_format

def split_dataset(fed_args, script_args, dataset):
    dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    elif fed_args.split_strategy == "no-iid":
        for i in range(fed_args.num_clients):
            category = fed_args.category_list[i]
            print(f"Client {i} - {category}")
            if not fed_args.whether_same_sample :
                filter_dataset = dataset.filter(lambda example: example["category"] == category)
            else :
                filter_dataset = dataset.filter(lambda example: example["category"] == category).select(range(100))
            local_datasets.append(filter_dataset)
    elif fed_args.split_strategy == "dirichlet":
        for i in range(fed_args.num_clients):
            category = fed_args.category_list[i]
            print(f"Client {i} - {category}")
            if not fed_args.whether_same_sample :
                filter_dataset = dataset.filter(lambda example: example["client_id"] == category)
            else :
                filter_dataset = dataset.filter(lambda example: example["client_id"] == category).select(range(100))
            local_datasets.append(filter_dataset) 

    return local_datasets

def get_dataset_this_round(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round

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
    else :
        print(f'NO THIS CATEGORY {category}')

    dataset = dataset.rename_column("context", "input")
    dataset = dataset.rename_column("response", "output")
    dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing for unified format.")

    assert len(dataset) == 100, \
        f"Expect {100}, but {len(dataset)}"

    return dataset