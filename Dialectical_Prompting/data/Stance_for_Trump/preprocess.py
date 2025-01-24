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
        "NONE": 0,
        "AGAINST": -1,
        "FAVOR": 1
    }
    return mapping.get(label, 0)  # Default to 0 if label is not found


def label_to_ans_text(label):

    mapping = {
        "NONE": "NEUTRAL",
        "AGAINST": "OPPOSITION",
        "FAVOR": "SUPPORT"
    }
    return mapping.get(label, "NEURAL")  

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
                "ans_label": label_to_ans_label(row["label"]),
                "text": row["text"],
                "ans_text": label_to_ans_text(row["label"])
            }
            tweets_list.append(tweet)

    with open(json_filename, mode='w') as f:
        json.dump(tweets_list, f, indent=4)

    print(f"CSV data has been successfully converted to JSON format and saved to {json_filename}")

# Example usage
csv_train_filename = 'trump_stance_train_public.csv'
json_train_filename = 'train.json'
csv_to_json(csv_train_filename, json_train_filename)
with open(json_train_filename, 'r') as f:
    data = json.load(f)
print("Train data size: ", len(data))


csv_test_filename = 'trump_stance_test_public.csv'
json_test_filename = 'val_and_test.json'
csv_to_json(csv_test_filename, json_test_filename)

# split val and test
with open(json_test_filename, 'r') as f:
    data = json.load(f)

val, test = train_test_split(data, test_size=0.7, random_state=42)


print("Validation data size: ", len(val))
print("Test data size: ", len(test))

with open('val.json', 'w') as file:
    json.dump(val, file, indent=4)

with open('test.json', 'w') as file:
    json.dump(test, file, indent=4)