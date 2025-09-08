import numpy as np
import os
from datasets import Dataset, load_dataset, concatenate_datasets 
from collections import defaultdict
from typing import List, Tuple

def split_dataset_for_fl(
    full_dataset: Dataset,
    allowed_categories: List[str],
    num_clients: int,
    test_set_size_per_category: int = 25,
    client_dataset_size: int = 1000,
    dirichlet_alpha: float = 1.0,
    seed: int = 42
) -> Tuple[Dataset, List[Dataset]]:
    """
    Splits a dataset for Federated Learning. Creates a central test set
    stratified by category and generates non-IID client datasets using
    a Dirichlet distribution over categories.

    Args:
        full_dataset: The input Hugging Face dataset (e.g., train split).
        allowed_categories: A list of categories to include in the split.
        num_clients: The desired number of clients.
        test_set_size_per_category: The number of samples per category
                                    for the central test set.
        client_dataset_size: The total number of samples for each client's dataset.
                             Clients' data can overlap.
        dirichlet_alpha: The alpha parameter for the Dirichlet distribution
                         to control non-IIDness (smaller alpha -> more non-IID).
        seed: Random seed for reproducibility.

    Returns:
        A tuple containing:
        - The central test dataset (Hugging Face Dataset).
        - A list of Hugging Face Datasets, one for each client.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    print(f"Filtering dataset for categories: {allowed_categories}")
    # Filter dataset by allowed categories
    filtered_dataset = full_dataset.filter(
        lambda example: example['category'] in allowed_categories
    )
    print(f"Filtered dataset size: {len(filtered_dataset)}")

    # Group data indices by category
    category_indices = defaultdict(list)
    for i, example in enumerate(filtered_dataset):
        category_indices[example['category']].append(i)

    print(f"Indices grouped for {len(category_indices)} categories.")

    # --- Create Central Test Set ---
    test_indices = []
    available_client_indices_by_category = {}

    for category in allowed_categories:
        indices = category_indices.get(category, [])
        num_available = len(indices)
        num_test_samples = min(test_set_size_per_category, num_available)

        if num_test_samples < test_set_size_per_category:
             print(f"Warning: Category '{category}' has only {num_available} samples, requested {test_set_size_per_category} for test set.")

        sampled_test_indices = np.random.choice(
            indices,
            size=num_test_samples,
            replace=False
        ).tolist()
        test_indices.extend(sampled_test_indices)

        available_client_indices = list(set(indices) - set(sampled_test_indices))
        available_client_indices_by_category[category] = available_client_indices

    central_test_dataset = filtered_dataset.select(test_indices)
    print(f"\nCentral test dataset created with {len(central_test_dataset)} samples.")

    # --- Create Client Datasets using Dirichlet Distribution ---
    print("\nGenerating client datasets...")

    all_available_indices_for_clients = [idx for sublist in available_client_indices_by_category.values() for idx in sublist]
    num_categories = len(allowed_categories)
    category_map = {allowed_categories[i]: i for i in range(num_categories)}

    dirichlet_probs = np.random.dirichlet(
        np.full(num_categories, dirichlet_alpha), num_clients
    )

    client_datasets = []
    for i in range(num_clients):
        client_indices = []
        weights = []
        for idx in all_available_indices_for_clients:
            category = filtered_dataset[int(idx)]['category']
            if category in category_map:
                 cat_idx = category_map[category]
                 weights.append(dirichlet_probs[i][cat_idx])
            else:
                 weights.append(0)

        weights_sum = sum(weights)
        if weights_sum == 0:
             print(f"Warning: No available samples for client {i+1} based on categories.")
             sampled_client_indices = []
        else:
            normalized_weights = np.array(weights) / weights_sum

            sampled_client_indices = np.random.choice(
                all_available_indices_for_clients,
                size=client_dataset_size,
                replace=True,
                p=normalized_weights
            ).tolist()

        client_indices.extend(sampled_client_indices)
        client_dataset = filtered_dataset.select(client_indices)
        client_datasets.append(client_dataset)
        print(f"Client {i+1} dataset generated with {len(client_dataset)} samples.")

    return central_test_dataset, client_datasets


if __name__ == "__main__":
    dataset_identifier = ''

    output_dir = './alpha0.5' 
    os.makedirs(output_dir, exist_ok=True) 

    try:
        print("Loading dataset...")
        full_dataset = load_dataset(dataset_identifier, split="train")
        print(f"Dataset '{dataset_identifier}' loaded successfully with {len(full_dataset)} samples.")

        allowed_categories = ['closed_qa', 'classification', 'information_extraction', 'summarization']
        num_clients_to_split = 10 

        print("\n--- Splitting dataset for FL ---")
        central_test_set, client_data_list = split_dataset_for_fl(
            full_dataset=full_dataset,
            allowed_categories=allowed_categories,
            num_clients=num_clients_to_split,
            test_set_size_per_category=25,
            client_dataset_size=1000,
            dirichlet_alpha=0.5,
            seed=42
        )

        print("\n--- Splitting complete, processing and saving combined training dataset as JSON ---")

        test_json_path = os.path.join(output_dir, 'central_test.json')
        print(f"Saving central test set to {test_json_path}...")
        central_test_set.to_json(test_json_path)
        print(f"Central test set saved.")

        client_datasets_with_id = []
        print("Adding 'client_id' column to each client dataset and preparing for concatenation...")
        for i, client_dataset in enumerate(client_data_list):
            client_id_str = f'client_{i+1}'

            client_dataset_with_id = client_dataset.add_column(
                'client_id',
                [client_id_str] * len(client_dataset) 
            )
            client_datasets_with_id.append(client_dataset_with_id)
            # print(f"Added 'client_id' column for client {i+1} ({client_id_str}).") 

        print(f"Concatenating {len(client_datasets_with_id)} client datasets...")
        combined_train_dataset = concatenate_datasets(client_datasets_with_id)
        print(f"Combined training dataset created with {len(combined_train_dataset)} samples.")

        combined_train_json_path = os.path.join(output_dir, 'combined_train.json') 
        print(f"Saving combined training dataset to {combined_train_json_path}...")

        combined_train_dataset.to_json(combined_train_json_path)
        print(f"Combined training dataset saved.")

        print("\n--- All datasets processed and saved as JSON files ---")


    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Attempted to save data to: {os.path.abspath(output_dir)}")