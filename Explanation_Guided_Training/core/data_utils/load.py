import os
import json
import torch
import csv
from data_utils.dataset import CustomDGLDataset


def load_gpt_preds(dataset, topk):
    preds = []
    fn = f'gpt_preds/{dataset}.csv'
    print(f"Loading topk preds from {fn}")
    with open(fn, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(int(value))
            preds.append(inner_list)

    pl = torch.zeros(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl


def load_data(dataset, use_gpt=False, mod='train'):
    from data_utils.load_general import get_raw_text_general as get_raw_text
    
    data = get_raw_text(use_gpt=use_gpt, dataset=dataset, mod=mod)


    return data
