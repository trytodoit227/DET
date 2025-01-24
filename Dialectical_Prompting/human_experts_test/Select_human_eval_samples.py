import os
import json
import random

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def random_select(data, num_samples=20, seed=42):
    samples = data["results"]
    random.seed(seed)
    select_samples = random.sample(samples, num_samples)
    return select_samples

def find_file(prefix, directory):
    for file_name in os.listdir(directory):
        if file_name.startswith(prefix) and file_name.endswith('.json'):
            return os.path.join(directory, file_name)
    return None

def process_datasets(datasets, input_dir):
    aggregated_results = []

    for dataset in datasets:
        
        file = find_file(dataset, input_dir)
        
        if not file:
            print(f'Files for {dataset} not found.')
            continue
        
        data = load_json(file)
        selected_samples = random_select(data, num_samples=20)
        for sample in selected_samples:
            sample['dataset'] = dataset
            aggregated_results.append(sample)
    num_samples = len(aggregated_results)
    print(f'Number of aggregated samples: {num_samples}')
    
    output_data = {
        'info': 'Aggregated human evaluated samples for all datasets',
        'results': aggregated_results
    }
    
    output_path = 'Aggregated_human_evaluated_samples.json'
    
    save_json(output_data, output_path)
    print(f'Aggregated human evaluated samples saved at {output_path}!')


datasets = [
    "Overruling-ourmethod",
    "StockEmotions-ourmethod",
    "Stance_for_Trump-ourmethod",
    "RTE-ourmethod",
    "ethos-ourmethod",
    "values-ourmethod",
]
input_dir = '../EQ-Evaluating-Input'

process_datasets(datasets, input_dir)