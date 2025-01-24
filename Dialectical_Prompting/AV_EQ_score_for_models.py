import os
import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def average_eq_score(data1, data2):
    avg_EQ_score = (data1['EQ_score']['EQ_score'] + data2['EQ_score']['EQ_score']) / 2
    return avg_EQ_score

def find_file(prefix, directory):
    for file_name in os.listdir(directory):
        if file_name.startswith(prefix) and file_name.endswith('.json'):
            return os.path.join(directory, file_name)
    return None

def process_datasets(datasets, model1, model2, output_dir, input_dir):
    aggregated_results = []

    for dataset in datasets:
        prefix1 = f"{dataset}-{model1}"
        prefix2 = f"{dataset}-{model2}"
        
        file1 = find_file(prefix1, input_dir)
        file2 = find_file(prefix2, input_dir)
        
        if not file1 or not file2:
            print(f'Files for {dataset} with models {model1} and {model2} not found.')
            continue
        
        data1 = load_json(file1)
        data2 = load_json(file2)
        
        avg_EQ_score = average_eq_score(data1, data2)
        
        result = {
            'dataset': dataset,
            'model1': model1,
            'model2': model2,
            'avg_EQ_score': avg_EQ_score,
        }
        aggregated_results.append(result)
    
    output_data = {
        'info': 'Aggregated EQ scores for all datasets',
        'results': aggregated_results
    }
    
    output_name = 'Aggregated_EQ_Scores.json'
    output_path = os.path.join(output_dir, output_name)
    
    save_json(output_data, output_path)
    print(f'Aggregated EQ scores saved at {output_path}!')

def extract_eq_scores(prefix, directory, output_path):
    eq_scores = []

    for file_name in os.listdir(directory):
        if file_name.startswith(prefix) and file_name.endswith('.json'):
            file_path = os.path.join(directory, file_name)
            data = load_json(file_path)
            eq_scores.append({
                'file_name': file_name,
                'EQ_score': data['EQ_score']['EQ_score']
            })

    output_data = {
        'info': f'EQ scores for files starting with {prefix}',
        'results': eq_scores
    }
    
    save_json(output_data, output_path)
    print(f'EQ scores saved at {output_path}!')

# Find ethos EQ scores
# prefix = 'ethos'
# input_dir = 'EQ-Evaluating-Output'
# output_path = 'EQ-Evaluating-Output/ethos_EQ_scores.json'

# extract_eq_scores(prefix, input_dir, output_path)


# averaged all EQscores
datasets = [
    "Overruling-ourmethod",
    "Overruling-gpt-3.5-turbo-explain-then-predict",
    "StockEmotions-ourmethod",
    "StockEmotions-gpt-3.5-turbo-explain-then-predict",
    "Stance_for_Trump-ourmethod",
    "Stance_for_Trump-gpt-3.5-turbo-explain-then-predict",
    "RTE-ourmethod",
    "RTE-gpt-3.5-turbo-explain-then-predict",
    "ethos-ourmethod",
    "ethos-gpt-3.5-turbo-explain-then-predict",
    "values-ourmethod",
    "values-gpt-3.5-turbo-explain-then-predict"
]
model1 = 'gpt-4o-2024-05-13'
model2 = 'claude-3-5-sonnet-20240620'
input_dir = 'EQ-Evaluating-Output'
output_dir = 'EQ-Evaluating-Output'  # Ensure this directory exists

process_datasets(datasets, model1, model2, output_dir, input_dir)