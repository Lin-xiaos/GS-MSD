## importing libraries
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import re
# import matplotlib.pyplot as plt

from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset
import torch
import csv
import os
import torch.nn.functional as F
from torch import nn


def readcsv(fileName):
    '''
    将所有对话的目标语句、上下文语句、讽刺标签、情感标签、情绪标签都集合到uttDict中
    目标语句和上下文语句都在utterance list中，以对话的时间顺序展开，最后是目标语句
    现在返回的utterance是所有的utt，最多的语句是12句，最少的语句是2句
    '''
    with open(fileName, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        uttNameList = []
        # context = np.array()
        for i in reader:
            if i[0] != '':
                uttNameList.append(i[0])
        uttNameList = list(set(uttNameList))
        uttNameList.remove("KEY")
        uttDict = {}
        for name in uttNameList:
            uttDict[name] = {}
            uttDict[name]['utterance'] = []
            uttDict[name]['sarcasm-label'] = []
            uttDict[name]['sentiment-label'] = []
            uttDict[name]['emotion-label'] = []
            uttDict[name]['utt-number'] = ''

    with open(fileName, 'r', encoding='utf-8') as f1:
        reader = csv.reader(f1)
        for item in reader:
            if item[0] == 'KEY' or item[0] == '':
                continue
            uttDict[item[0]]['sarcasm-label'].append(item[4])
            uttDict[item[0]]['sentiment-label'].append(item[5])
            uttDict[item[0]]['emotion-label'].append(item[7])
            uttDict[item[0]]['utterance'].append(item[2])
            uttDict[item[0]]['utt-number'] = item[0]
    return uttDict, uttNameList


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # removes links
    text = re.sub(r'(?P<url>https?://[^\s]+)', r'', text)

    # remove @usernames
    text = re.sub(r"\@(\w+)", "", text)

    # remove # from #tags
    text = text.replace('#', '')

    return text


class TextDataset(Dataset):
    def __init__(self, uttDict, uttNameList, tokenizer):
        self.uttDict = uttDict
        self.tokenizer = tokenizer
        self.keys = uttNameList

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        key = self.keys[idx]
        text_list = self.uttDict[key]['utterance']

        input_ids_list = []
        attention_mask_list = []
        full_input_ids, full_attention_mask = extract_text_features(str(text_list), self.tokenizer)
        for text in text_list:
            input_ids, attention_mask = extract_text_features(text, self.tokenizer)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'full_input_ids': full_input_ids,
            'full_attention_mask': full_attention_mask,
            'unique_id': key
        }


def extract_text_features(text, tokenizer):
    encoded_sent = tokenizer.encode_plus(
        text=text_preprocessing(str(text)),  # Preprocess sentence
        add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
        max_length=128,  # Max length to truncate/pad
        padding='max_length',  # Pad sentence to max length
        return_attention_mask=True,  # Return attention mask
        truncation=True
    )

    input_ids = encoded_sent.get('input_ids')
    attention_mask = encoded_sent.get('attention_mask')

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)

    return input_ids, attention_mask


def store_data(bert, device, uttNameList, dataset, root_dir, store_dir):
    lengths = []
    bert.eval()

    for idx in tqdm(range(len(uttNameList))):
        sample = dataset.__getitem__(idx)

        unique_id = sample['unique_id']
        input_ids_list = sample['input_ids']
        attention_mask_list = sample['attention_mask']
        full_input_ids = sample['full_input_ids']
        full_attention_mask = sample['full_attention_mask']
        with torch.no_grad():
            full_out = bert(input_ids=full_input_ids.unsqueeze(0).to(device), attention_mask=full_attention_mask.unsqueeze(0).to(device))
        full_out_cls = full_out.last_hidden_state[:, 0, :].unsqueeze(0).detach().cpu().squeeze(0).numpy()  # cls vector
        filename = f'{root_dir}{store_dir}/{unique_id}_full_dialogue.npy'
        np.save(filename, full_out_cls)

        for i, (input_ids, attention_mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)

            num_tokens = attention_mask.sum().detach().cpu().item()

            with torch.no_grad():
                out = bert(input_ids=input_ids, attention_mask=attention_mask)

            out_tokens = out.last_hidden_state[:, 1:num_tokens, :].detach().cpu().squeeze(0).numpy()  # token vectors

            # Save token-level representations
            filename = f'{root_dir}{store_dir}/{unique_id}_{i}.npy'
            np.save(filename, out_tokens)

            lengths.append(num_tokens)

            # Save semantic/whole text representation
            out_cls = out.last_hidden_state[:, 0, :].unsqueeze(0).detach().cpu().squeeze(0).numpy()  # cls vector
            filename = f'{root_dir}{store_dir}/{unique_id}_{i}_full_text.npy'
            np.save(filename, out_cls)

    return lengths


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Base directory for the data
    root_dir = "/home/zxl/MultiTask classification/"

    ## File locations
    train_csv_name = "/home/zxl/MultiTask classification/M2Seq2Seq-master/mustard-dataset-train.csv"
    test_csv_name = "/home/zxl/MultiTask classification/M2Seq2Seq-master/mustard-dataset-test.csv"
    dev_csv_name = "/home/zxl/MultiTask classification/M2Seq2Seq-master/mustard-dataset-dev.csv"

    ## Loading model and tokenizer from HuggingFace
    tokenizer = BertTokenizer.from_pretrained(
        "/home/zxl/MultiTask classification/MultiTask-Classfication/bert-base-uncased/",
        do_lower_case=True)  ## ## bert-base-uncased - for english dataset
    bert = BertModel.from_pretrained("/home/zxl/MultiTask classification/MultiTask-Classfication/bert-base-uncased/",
                                     return_dict=True)

    bert = bert.to(device)

    ## Directory to store the node embeddings for each post
    store_dir = "GAME-ON-main/mustard/textfeature/"

    ## Create graph data for training set
    # df_train = pd.read_csv(f'{train_csv_name}', encoding='utf-8')
    # df_train = df_train.dropna().reset_index(drop=True)
    # Assume uttDict is obtained from readcsv and processUttDict functions
    uttDict, uttNameList = readcsv(train_csv_name)
    train_dataset = TextDataset(uttDict, uttNameList, tokenizer)

    lengths = store_data(bert, device, uttNameList, train_dataset, root_dir, store_dir)

    ## Create graph data for testing set
    # df_test = pd.read_csv(f'{test_csv_name}', encoding='utf-8')
    uttDict, uttNameList = readcsv(test_csv_name)
    test_dataset = TextDataset(uttDict, uttNameList, tokenizer)

    lengths = store_data(bert, device, uttNameList, test_dataset, root_dir, store_dir)

    ## Create graph data for dev set
    # df_dev = pd.read_csv(f'{dev_csv_name}', encoding='utf-8')
    uttDict, uttNameList = readcsv(dev_csv_name)
    dev_dataset = TextDataset(uttDict, uttNameList, tokenizer)

    lengths = store_data(bert, device, uttNameList, dev_dataset, root_dir, store_dir)

