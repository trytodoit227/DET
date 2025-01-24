import csv
import json
from sklearn.model_selection import train_test_split
import random

def label_to_ans_text(label):

    mapping = {
        1 : "overruling",
        0 : "non_overruling"
    }
    return mapping.get(label, 0)  # Default to 0 if label is not found

def csv_to_json(csv_filename, json_filename):
    """
    Convert a CSV file to a JSON file.
    
    :param csv_filename: Input CSV file name
    :param json_filename: Output JSON file name
    """
    tweets_list = []
    with open(csv_filename, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            tweet = {
                "ans_label": int(row["label"]),
                "text": row["sentence1"],
                "ans_text": label_to_ans_text(int(row["label"]))
            }
            tweets_list.append(tweet)
        random.shuffle(tweets_list)

    with open(json_filename, mode='w') as f:
        json.dump(tweets_list, f, indent=4)

    print(f"CSV data has been successfully converted to JSON format and saved to {json_filename}")

# Example usage
csv_train_filename = 'overruling.csv'
json_train_filename = 'overruling.json'


csv_to_json(csv_train_filename, json_train_filename)

# split train, val and test
with open(json_train_filename, 'r') as f:
    data = json.load(f)

train_data, val_test_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(val_test_data, test_size=0.65, random_state=42)

# save data
with open('train.json', mode='w') as f:
    json.dump(train_data, f, indent=4)
with open('val.json', mode='w') as f:
    json.dump(val_data, f, indent=4)
with open('test.json', mode='w') as f:
    json.dump(test_data, f, indent=4)

print("Train data size: ", len(train_data))
print("Val data size: ", len(val_data))
print("Test data size: ", len(test_data))