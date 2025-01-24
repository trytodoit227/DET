import csv
import json
from sklearn.model_selection import train_test_split

def label_to_ans_label(label):
    """
    Convert label to ans_label.
    NONE: 0
    AGAINST: -1
    FAVOR: 1
    """
    mapping = {
        "bearish": 0,
        "bullish": 1
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
                "ans_label": label_to_ans_label(row["senti_label"]),
                "text": row["processed"],
                "ans_text": row["senti_label"]
            }
            tweets_list.append(tweet)

    with open(json_filename, mode='w') as f:
        json.dump(tweets_list, f, indent=4)

    print(f"CSV data has been successfully converted to JSON format and saved to {json_filename}")

# Example usage
csv_train_filename = 'train_stockemo.csv'
json_train_filename = 'train.json'
csv_val_filename = 'val_stockemo.csv'
json_val_filename = 'val.json'
csv_test_filename = 'test_stockemo.csv'
json_test_filename = 'test.json'


csv_to_json(csv_train_filename, json_train_filename)
csv_to_json(csv_val_filename, json_val_filename)
csv_to_json(csv_test_filename, json_test_filename)

with open(json_train_filename, 'r') as f:
    data = json.load(f)
print("Train data size: ", len(data))
with open(json_val_filename, 'r') as f:
    data = json.load(f)
print("Val data size: ", len(data))
with open(json_test_filename, 'r') as f:
    data = json.load(f)
print("Test data size: ", len(data))
