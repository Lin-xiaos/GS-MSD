## importing libraries
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import re
import json
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torch import nn
import spacy
nlp = spacy.load("en_core_web_sm")


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

def extract_dependency_tree(text):
    """
    使用spaCy提取文本的依存关系树。
    :param text: 需要分析的文本。
    :return: 依存树的结构化信息（字典形式）
    """
    if isinstance(text, float):
        # 如果是浮点数，将其转换为字符串，或者处理缺失值
        if pd.isna(text):
            text = ""  # 处理缺失值，将其转换为空字符串
        else:
            text = str(text)
    doc = nlp(text)  # Process the text using spaCy

    # 构建依存树：包括每个词及其父词和依赖关系类型
    dependency_tree = []
    for token in doc:
        dependency_tree.append({
            'text': token.text,
            'dep': token.dep_,
            'head': token.head.text,
            'head_pos': token.head.pos_,
            'pos': token.pos_
        })

    return dependency_tree

class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        ## Main DataFrame with all the tweets
        self.df = df.reset_index(drop=True)

        ## TOkenizer to be used
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## Textual content of the post
        text = self.df['text_corrected'][idx]

        ## Unique id to be used as identifier
        unique_id = self.df['image_name'][idx].split(".")[0]

        # Create empty lists to store outputs
        input_ids = []
        attention_mask = []

        encoded_sent = self.tokenizer.encode_plus(
            text=text_preprocessing(str(text)),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=512,  # Max length to truncate/pad
            padding='max_length',  # Pad sentence to max length
            return_attention_mask=True,  # Return attention mask
            truncation=True
        )

        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'unique_id': unique_id}


def store_data(bert, device, df, dataset, store_dir):
    lengths = []
    bert.eval()

    for idx in tqdm(range(len(df))):
        sample = dataset.__getitem__(idx)

        input_ids, attention_mask = sample['input_ids'].unsqueeze(0), sample['attention_mask'].unsqueeze(0)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        unique_id = sample['unique_id']

        num_tokens = attention_mask.sum().detach().cpu().item()

        with torch.no_grad():
            out = bert(input_ids=input_ids, attention_mask=attention_mask)

        out_tokens = out.last_hidden_state[:, 1:num_tokens, :].detach().cpu().squeeze(0).numpy()  ## token vectors

        ## Save token-level representations
        filename = f'{root_dir}{store_dir}{unique_id}.npy'
        np.save(filename, out_tokens)

        lengths.append(num_tokens)

        ## Save semantic/ whole text representation
        out_cls = out.last_hidden_state[:, 0, :].unsqueeze(0).detach().cpu().squeeze(0).numpy()  ## cls vector
        filename = f'{root_dir}{store_dir}{unique_id}_full_text.npy'
        np.save(filename, out_cls)

        # Extract dependency tree for the text and save it
        text = df['text_corrected'][idx]
        dep_tree = extract_dependency_tree(text)

        # Save the dependency tree as a JSON file
        dep_filename = f'{root_dir}{store_dir}{unique_id}_dep_tree.json'
        with open(dep_filename, 'w', encoding='utf-8') as dep_file:
            json.dump(dep_tree, dep_file, ensure_ascii=False, indent=4)

    return lengths


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Base directory for the data
    root_dir = "/home/zxl/MultiTask classification/"

    # File locations
    train_csv_name = "data/train2.csv"
    test_csv_name = "data/test2.csv"
    dev_csv_name = "data/dev2.csv"

    # Loading model and tokenizer from HuggingFace
    tokenizer = BertTokenizer.from_pretrained(
        "/home/zxl/MultiTask classification/MultiTask-Classfication/bert-base-uncased/",
        do_lower_case=True)  ## ## bert-base-uncased - for english dataset
    bert = BertModel.from_pretrained("/home/zxl/MultiTask classification/MultiTask-Classfication/bert-base-uncased/",
                                     return_dict=True)

    bert = bert.to(device)

    # Directory to store the node embeddings for each post
    store_dir = "proposed/test2/large-model/memotion/Textfeature/"

    # Create graph data for training set
    df_train = pd.read_csv(f'{root_dir}{train_csv_name}', encoding='latin')
    # df_train = df_train.dropna().reset_index(drop=True)
    train_dataset = TextDataset(df_train, tokenizer)

    lengths = store_data(bert, device, df_train, train_dataset, store_dir)

    # Create graph data for testing set
    df_test = pd.read_csv(f'{root_dir}{test_csv_name}', encoding='latin')
    # df_test = df_test.dropna().reset_index(drop=True)
    test_dataset = TextDataset(df_test, tokenizer)

    lengths = store_data(bert, device, df_test, test_dataset, store_dir)

    # Create graph data for valid set
    df_dev = pd.read_csv(f'{root_dir}{dev_csv_name}', encoding='latin')
    # df_test = df_test.dropna().reset_index(drop=True)
    dev_dataset = TextDataset(df_dev, tokenizer)

    lengths = store_data(bert, device, df_dev, dev_dataset, store_dir)
