import json

def convert_jsonl_to_json(input_file, output_file):
    # Read input JSONL file
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    # List to hold all processed data
    processed_data = []
    
    for line in lines:
        # Parse each line as JSON
        original_data = json.loads(line)
        
        # Convert to new format
        new_data = {
            "ans_label": 0 if original_data["label_text"] == "not entailment" else 1,
            "text": f'PREMISE:{original_data["text1"]}\nHYPOTHESIS: {original_data["text2"]}',
            "ans_text": original_data["label_text"]
        }
        
        # Add to the list of processed data
        processed_data.append(new_data)
    
    # Write the processed data to the output JSON file
    with open(output_file, 'w') as outfile:
        json.dump(processed_data, outfile, indent=4)

    print(f"Data has been successfully processed and written to {output_file}")

# Call the function
input_file = 'train_setfit.jsonl'
output_file = 'train_setfit.json'
convert_jsonl_to_json(input_file, output_file)

input_file = 'validation_setfit.jsonl'
output_file = 'validation_setfit.json'
convert_jsonl_to_json(input_file, output_file)

# import json

# def load_json(file_path):
#     with open(file_path, 'r') as file:
#         return json.load(file)

# def find_common_samples(file1, file2):
#     data1 = load_json(file1)
#     data2 = load_json(file2)
    
#     # Convert lists of dictionaries to sets of tuples to easily find common elements
#     set1 = {tuple(sample.items()) for sample in data1}
#     set2 = {tuple(sample.items()) for sample in data2}
    
#     common_samples = set1.intersection(set2)
    
#     return len(common_samples), common_samples

# file1 = 'train_setfit.json'
# file2 = 'val.json'

# common_count, common_samples = find_common_samples(file1, file2)

# print(f"Number of common samples: {common_count}")
# Optionally print common samples
# for sample in common_samples:
#     print(dict(sample))