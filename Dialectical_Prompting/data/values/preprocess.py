import pandas as pd
import json
import random
from sklearn.model_selection import train_test_split

def preprocess_to_json(csv_file_path: str, json_file_path: str):
    df = pd.read_csv(csv_file_path)
    
    # Filter the DataFrame to only include rows where 'orig_label' is -1 or 1
    # relevant_df =  df.loc[df['orig_label'].isin([-1, 1])]

    # df_copy = relevant_df.copy()
    # df_copy.loc[df_copy['orig_label'] == -1, 'orig_label'] = 0
    # relevant_df = df_copy

    relevant_df = df.copy()

    # Map 'orig_label' to 'ans_text' with 'entailment' for 1 and 'not entailment' for 0
    relevant_df['ans_text'] = relevant_df['orig_label'].map({1: 'AFFIRMATION', 0: 'IRRELEVANCE', -1:'REJECTION'})
    
    relevant_df = relevant_df[['scenario', 'orig_label', 'ans_text']]
    relevant_df = relevant_df.rename(columns={'scenario': 'text', 'orig_label': 'ans_label'})
    relevant_dict = relevant_df.to_dict(orient='records')

    # Save the list of dictionaries to a JSON file
    with open(json_file_path, 'w') as f:
        json.dump(relevant_dict, f, indent=4)

def select_random_samples(json_file_path: str, num_samples: int):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Select random samples
    random_samples = random.sample(data, num_samples)
    
    return random_samples

def save_samples_to_json(samples: list, json_file_path: str):
    with open(json_file_path, 'w') as f:
        json.dump(samples, f, indent=4)

def count_samples_in_json(json_file_path: str):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return len(data)

def data_reform_value(json_file_path: str):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    reform_data = []
    with open("value_sentence_reformat.txt", "r") as f:
        template = f.read()
    for sample in data:
        parts = sample['text'].lstrip('[').split('] ', 1)
        prompt = template.format(
                            special_value=parts[0],
                            special_text=parts[1],
                            )
        new_sample = {}
        new_sample['text'] = prompt
        new_sample['ans_label'] = sample['ans_label']
        new_sample['ans_text'] = sample['ans_text']
        reform_data.append(new_sample)
        
    return reform_data

# Use the function
preprocess_to_json('Valuenet_original_test.csv', 'complete_test.json')
preprocess_to_json('Valuenet_original_val.csv', 'complete_val.json')
preprocess_to_json('Valuenet_original_train.csv', 'complete_train.json')


# Use the function data_reform_value and save the reformatted data to a new JSON file
reform_train = data_reform_value('complete_train.json')
reform_val = data_reform_value('complete_val.json')
reform_test = data_reform_value('complete_test.json')
with open('train.json', 'w') as f:
    json.dump(reform_train, f, indent=4)
with open('val.json', 'w') as f:
    json.dump(reform_val, f, indent=4)
with open('test.json', 'w') as f:
    json.dump(reform_test, f, indent=4)


print("Train data size: ", len(reform_train))
print("Val data size: ", len(reform_val))
print("Test data size: ", len(reform_test))

# # Use the functions select_random_samples and save_samples_to_json
# random_samples_train = select_random_samples('train.json', 600)
# save_samples_to_json(random_samples_train, 'train.json')

# num_samples_train = count_samples_in_json('train.json')
# print(f"Number of samples in train.json: {num_samples_train}")

# # random_samples_val = select_random_samples('complete_val.json', num_samples_complete_val)
# # save_samples_to_json(random_samples_val, 'val.json')

# # num_samples_val = count_samples_in_json('val.json')
# # print(f"Number of samples in val.json: {num_samples_val}")

# random_samples_test = select_random_samples('test.json',300)
# save_samples_to_json(random_samples_test, 'test.json')

# num_samples_test = count_samples_in_json('test.json')
# print(f"Number of samples in test.json: {num_samples_test}")