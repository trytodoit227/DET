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

def process_datasets(datasets, input_dir, model):
    aggregated_results = []

    # if model is claude, we need to reduce ethos dataset in the dataset list
    if model == "claude":
        datasets.remove("ethos-ourmethod")

    for dataset in datasets:

        filename = dataset + "-" + model
        
        file = find_file(filename, input_dir)
        
        if not file:
            print(f'Files for {filename} not found.')
            continue
        
        data = load_json(file)
        reference_file = 'Aggregated_human_evaluated_samples.json'
        reference_data = load_json(reference_file)

        # look for the same samples in the reference data
        samples = []
        for sample in data['results']:
            for ref_sample in reference_data['results']:
                if sample["original_data"]['text'] == ref_sample["original_data"]['text']:
                    samples.append(sample)
                    break
        # add the selected samples to the aggregated results
        aggregated_results.extend(samples)
        print(f'Number of samples for {dataset}: {len(samples)}')

    num_samples = len(aggregated_results)
    print(f'Number of aggregated samples: {num_samples}')
    
    output_data = {
        'info': 'Aggregated machine evaluated samples for all datasets',
        'results': aggregated_results
    }
    
    output_path = "Aggregated_machine_samples_" + model + ".json"
    
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
model1 = "gpt-4o"
model2 = "claude"
input_dir = '../EQ-Evaluating-Output'

process_datasets(datasets, input_dir, model1)
process_datasets(datasets, input_dir, model2)