import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import re
import spacy  # Import spaCy for dependency parsing
import json
from torch.utils.data import Dataset
import torch
import csv
from transformers import BertModel, BertTokenizer
from torch import nn

# Load spaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")


def readcsv(fileName):
    '''
    将所有对话的目标语句、上下文语句、讽刺标签、情感标签、情绪标签都集合到uttDict中
    目标语句和上下文语句都在utterance list中，以对话的时间顺序展开，最后是目标语句
    现在返回的utterance是所有的utt，最多的语句是12句，最少的语句是2句
    '''
    with open(fileName, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        uttNameList = []
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


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out


# Function to perform dependency parsing on a sentence
def get_dependency_tree(sentence):
    doc = nlp(sentence)
    dep_tree = []
    for token in doc:
        dep_tree.append({
            'text': token.text,
            'dep': token.dep_,
            'head': token.head.text,
            'pos': token.pos_,
            'tag': token.tag_
        })
    return dep_tree


def create_dependency_graph(uttDict):
    """
    创建依存图，基于每个句子的依存树和句子间的关系来构建图。
    """
    dependency_graph = {}
    for utt_name, utt_info in uttDict.items():
        sentences = utt_info['utterance']
        utterance_graph = []

        # Process each sentence in the utterance
        for sentence in sentences:
            dep_tree = get_dependency_tree(sentence)
            utterance_graph.append(dep_tree)

        # Store the graph for each utterance
        dependency_graph[utt_name] = utterance_graph
    return dependency_graph


class TextDataset(Dataset):
    def __init__(self, uttDict, uttNameList, tokenizer, dependency_graph):
        self.uttDict = uttDict
        self.tokenizer = tokenizer
        self.keys = uttNameList
        self.dependency_graph = dependency_graph

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        key = self.keys[idx]
        text_list = self.uttDict[key]['utterance']

        input_ids_list = []
        attention_mask_list = []

        # Retrieve dependency tree for each sentence
        dep_tree_list = self.dependency_graph[key]

        for text, dep_tree in zip(text_list, dep_tree_list):
            input_ids, attention_mask = extract_text_features(text, self.tokenizer)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'unique_id': key,
            'dep_tree': dep_tree_list  # Include dependency tree in the output
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


def store_data(bert, bilstm, device, uttNameList, dataset, root_dir, store_dir):
    lengths = []
    bert.eval()

    for idx in tqdm(range(len(uttNameList))):
        sample = dataset.__getitem__(idx)

        unique_id = sample['unique_id']
        input_ids_list = sample['input_ids']
        attention_mask_list = sample['attention_mask']
        dep_tree_list = sample['dep_tree']

        for i, (input_ids, attention_mask, dep_tree) in enumerate(zip(input_ids_list, attention_mask_list, dep_tree_list)):
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)

            num_tokens = attention_mask.sum().detach().cpu().item()

            with torch.no_grad():
                out = bert(input_ids=input_ids, attention_mask=attention_mask)

            out_tokens = out.last_hidden_state[:, 1:num_tokens, :].detach().cpu().squeeze(0).to(device)  # token vectors
            out_tokens = torch.tensor(out_tokens, dtype=torch.float32).unsqueeze(0).to(
                device)  # (1, seq_len, hidden_dim)

            # 送入 BiLSTM 处理
            bilstm_out = bilstm(out_tokens)  # (batch=1, seq_len, hidden_dim * 2)

            token_features = bilstm_out.squeeze(0).detach().cpu().numpy()
            sentence_feature = bilstm_out[:, -1, :].squeeze(0).detach().cpu().numpy()

            # Save token-level representations
            filename = f'{root_dir}{store_dir}/{unique_id}_{i}.npy'
            np.save(filename, token_features)

            lengths.append(num_tokens)
            print(f"bilstm_out.shape: {bilstm_out.shape}")

            # Save semantic/whole text representation
            filename = f'{root_dir}{store_dir}/{unique_id}_{i}_full_text.npy'
            np.save(filename, sentence_feature)

    return lengths


def save_dependency_graph_to_json(dependency_graph, output_file):
    """
    将依存图保存为 JSON 文件。
    :param dependency_graph: 依存图数据（字典形式）
    :param output_file: 输出文件路径
    """
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dependency_graph, f, indent=4)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Base directory for the data
    root_dir = "/home/zxl/MultiTask classification/"

    # File locations
    train_csv_name = "/home/zxl/MultiTask classification/M2Seq2Seq-master/mustard-dataset-train.csv"
    test_csv_name = "/home/zxl/MultiTask classification/M2Seq2Seq-master/mustard-dataset-test.csv"
    dev_csv_name = "/home/zxl/MultiTask classification/M2Seq2Seq-master/mustard-dataset-dev.csv"

    # Loading model and tokenizer from HuggingFace
    tokenizer = BertTokenizer.from_pretrained(
        "/home/zxl/MultiTask classification/MultiTask-Classfication/bert-base-uncased/",
        do_lower_case=True)  ## ## bert-base-uncased - for english dataset
    bert = BertModel.from_pretrained("/home/zxl/MultiTask classification/MultiTask-Classfication/bert-base-uncased/",
                                     return_dict=True)

    model = bert.to(device)
    hidden_dim = 256  # 隐藏层维度
    input_dim = 768  # BERT 输出的维度
    bilstm = BiLSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    # Directory to store the node embeddings for each post
    store_dir = "proposed/test2/large-model/Textfeature/"

    # Create graph data for training set
    uttDict, uttNameList = readcsv(train_csv_name)
    dependency_graph = create_dependency_graph(uttDict)
    # print(dependency_graph)
    train_dataset = TextDataset(uttDict, uttNameList, tokenizer, dependency_graph)

    # save_dependency_graph_to_json(dependency_graph, store_dir)
    # save_dependency_graph_to_json(dependency_graph,
    # "/home/zxl/MultiTask classification/proposed/test2/large-model/mustard/Textfeature/dependency_graph_train.json")
    lengths = store_data(model, bilstm, device, uttNameList, train_dataset, root_dir, store_dir)

    # Create graph data for testing set
    uttDict, uttNameList = readcsv(test_csv_name)
    dependency_graph = create_dependency_graph(uttDict)
    test_dataset = TextDataset(uttDict, uttNameList, tokenizer, dependency_graph)
    # save_dependency_graph_to_json(dependency_graph,
    #                               store_dir)
    # save_dependency_graph_to_json(dependency_graph,
    # "/home/zxl/MultiTask classification/proposed/test2/large-model/mustard/Textfeature/dependency_graph_test.json")

    lengths = store_data(model, bilstm, device, uttNameList, test_dataset, root_dir, store_dir)

    # Create graph data for dev set
    uttDict, uttNameList = readcsv(dev_csv_name)
    dependency_graph = create_dependency_graph(uttDict)
    dev_dataset = TextDataset(uttDict, uttNameList, tokenizer, dependency_graph)
    # save_dependency_graph_to_json(dependency_graph,
    #                               store_dir)
    # save_dependency_graph_to_json(dependency_graph,
    # "/home/zxl/MultiTask classification/proposed/test2/large-model/mustard/Textfeature/dependency_graph_dev.json")

    lengths = store_data(model, bilstm, device, uttNameList, dev_dataset, root_dir, store_dir)



