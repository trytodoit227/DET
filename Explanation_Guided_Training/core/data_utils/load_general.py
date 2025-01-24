import torch
import pandas as pd
import json


def get_raw_text_general(use_gpt=False, dataset='ethos', mod='train'):
    #for values
    with open(f'/home/swufe/DHM/DET/Explanation_Guided_Training/dataset/{dataset}/exp-gpt-3.5-{mod}.json', encoding='utf-8') as f:
        orig_data = json.load(f)

    #for other data
    # with open(f'/home/swufe/DHM/DET/Explanation_Guided_Training/dataset/{dataset}/exp-{mod}.json', encoding='utf-8') as f:
    #     orig_data = json.load(f)


    # Get the label list
    label_list = orig_data['info']['dataset']['data_label_list']

    # Create a new dataset
    data = []

    for result in orig_data['results']:
        # Get the original text and predicted result
        original_text = result['original_data']['text']
        # Convert ans_text to a numeric label
        original_label = result['original_data']['ans_text']
        numeric_label = label_list.index(original_label)
        if use_gpt:
            pred_result = result['pred_result']
            # Dynamically combine the new text string
            new_text = f"Input Text: {original_text}\n"
            # new_text =""
            for key, value in pred_result.items():
                if key.startswith('why_'):
                    new_text += f"{key.replace('_', ' ').title()}: {value}\n"
                # if key.startswith('why_') or key.startswith('pred'):
                #     new_text += f"{key.replace('_', ' ').title()}: {value}\n"                
                # new_text += f"{key.replace('_', ' ').title()}: {value}\n"
            # new_text += f"Original Text: {original_text}\n"
            # Add the new text string and numeric label to the new dataset
            data.append({
                'text': new_text.strip(),  # Remove trailing newline
                'label': numeric_label
            })
        else:
            data.append({
                'text': original_text,
                'label': numeric_label
            })

    return data