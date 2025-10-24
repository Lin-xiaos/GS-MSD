import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

## Importing libraries
import numpy as np
import pandas as pd
import math
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import conv

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
import csv
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
import config, model_mu, dataset, engine, utils

# import wandb

# wandb.init(project="", entity="", name="")

def readcsv(fileName):
    '''
    将所有对话的目标语句、上下文语句、讽刺标签、情感标签、情绪标签都集合到uttDict中
    目标语句和上下文语句都在utterance list中，以对话的时间顺序展开，最后是目标语句
    现在返回的utterance是所有的utt，最多的语句是12句，最少的语句是2句
    '''
    with open(fileName, 'r', encoding='latin') as f:
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

    with open(fileName, 'r', encoding='latin') as f1:
        reader = csv.reader(f1)
        for item in reader:
            if item[0] == 'KEY' or item[0] == '':
                continue
            uttDict[item[0]]['sarcasm-label'].append(item[4])
            uttDict[item[0]]['sentiment-label'] .append(item[5])
            uttDict[item[0]]['emotion-label'].append(item[7])
            uttDict[item[0]]['utterance'].append(item[2])
            uttDict[item[0]]['utt-number'] = item[0]
    return uttDict, uttNameList


if __name__ == '__main__':
    # Setup the dataset
    dataset_name = "mu"  ## me15, we
    utils.set_seed(5)

    if dataset_name == "me":
        dataset_train, dataset_test, dataset_dev = utils.set_up_memotion()
    elif dataset_name == "mu":
        dataset_train, dataset_test, dataset_dev = utils.set_up_mustard()
    else:
        print("No Data")

    # Setup the dataloaders
    dataloader_train = GraphDataLoader(
        dataset_train,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=True)

    dataloader_test = GraphDataLoader(
        dataset_test,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False)

    dataloader_dev = GraphDataLoader(
        dataset_dev,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Inititalize the GAME-ON model
    model = model_mu.MModel()

    # FInd the number of parameters
    print("Total number of parameters:", sum(p.numel() for p in model.parameters()))

    model.to(device)

    # Calculate number of train steps
    num_update_steps_per_epoch = math.ceil(len(dataloader_train) / config.gradient_accumulation_steps)
    num_train_steps = num_update_steps_per_epoch * config.epochs

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=num_train_steps
    )

    # model.load_state_dict(torch.load("/home/zxl/MultiTask classification/proposed/test2/memotion/best_model.pth"))
    model.load_state_dict(torch.load("/home/zxl/MultiTask classification/proposed/test2/mustard/best_model.pth"))

    model.eval()

    datafile = "/home/zxl/MultiTask classification/M2Seq2Seq-master/mustard-dataset-test.csv"
    uttDict, uttNameList = readcsv(datafile)
    idx2uttName = {idx: uttNameList[idx] for idx in range(len(uttNameList))}

    all_utt_names = []
    all_preds = []
    predictions_dict = {}

    with torch.no_grad():
        for batch in tqdm(dataloader_test):
            batched_graph, text_graph, image_graph, audio_graph, _, _, indices = batch

            batched_graph = batched_graph.to(device)
            text_graph = text_graph.to(device)
            image_graph = image_graph.to(device)
            audio_graph = audio_graph.to(device)

            sarcasm_logits, sentiment, _ = model(batched_graph, text_graph, image_graph, audio_graph)
            preds = torch.argmax(sentiment, dim=1).cpu().numpy()
            batch_indices = indices.cpu().numpy()

            # 将 batch 中每个样本的索引转换为对话名称（KEY）
            for i, idx in enumerate(batch_indices):
                key = idx2uttName[idx]  # 得到当前对话对应的 KEY
                # 如果还没有该对话的预测，则存入；如果已有，则跳过该对话
                if key not in predictions_dict:
                    predictions_dict[key] = preds[i]

    # 转成 DataFrame 并按 index 对齐
    prediction_df = pd.DataFrame(list(predictions_dict.items()), columns=['KEY', 'sentiment_pred'])

    # 读取原始测试数据
    test_df = pd.read_csv(datafile, encoding='latin')
    test_df.columns = test_df.columns.str.strip()  # 清理列名

    # 合并原始数据和预测标签（确保每个对话只有一个预测标签）
    test_df = test_df.merge(prediction_df, on='KEY', how='left')

    # 可选：保存结果
    # test_df.to_csv("/home/zxl/MultiTask classification/proposed/test2/memotion/test_with_predictions.csv", index=False)
    test_df.to_csv("/home/zxl/MultiTask classification/proposed/test2/mustard/test_with_predictions1.csv", index=False)
    print(f"✅ 每个对话的讽刺预测标签已写入每句话中")


# if __name__ == '__main__':
#     # Setup the dataset
#     dataset_name = "me"  ## me15, we
#     utils.set_seed(5)
#
#     if dataset_name == "me":
#         dataset_train, dataset_test, dataset_dev = utils.set_up_memotion()
#     elif dataset_name == "mu":
#         dataset_train, dataset_test, dataset_dev = utils.set_up_mustard()
#     else:
#         print("No Data")
#
#     # Setup the dataloaders
#     dataloader_train = GraphDataLoader(
#         dataset_train,
#         batch_size=config.batch_size,
#         drop_last=False,
#         shuffle=True)
#
#     dataloader_test = GraphDataLoader(
#         dataset_test,
#         batch_size=config.batch_size,
#         drop_last=False,
#         shuffle=False)
#
#     dataloader_dev = GraphDataLoader(
#         dataset_dev,
#         batch_size=config.batch_size,
#         drop_last=False,
#         shuffle=False)
#
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#
#     # Inititalize the GAME-ON model
#     model = model_mu.MModel()
#
#     # FInd the number of parameters
#     print("Total number of parameters:", sum(p.numel() for p in model.parameters()))
#
#     model.to(device)
#
#     # Calculate number of train steps
#     num_update_steps_per_epoch = math.ceil(len(dataloader_train) / config.gradient_accumulation_steps)
#     num_train_steps = num_update_steps_per_epoch * config.epochs
#
#     optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
#
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer, num_warmup_steps=5, num_training_steps=num_train_steps
#     )
#
#     # model.load_state_dict(torch.load("/home/zxl/MultiTask classification/proposed/test2/memotion/best_model.pth"))
#     model.load_state_dict(torch.load("/home/zxl/MultiTask classification/proposed/test2/mustard/best_model.pth"))
#
#     model.eval()
#
#     # 准备记录预测结果
#     all_indices = []
#     all_preds = []
#
#     with torch.no_grad():
#         for batch in tqdm(dataloader_test):
#             batched_graph, text_graph, image_graph, _, _, indices = batch
#
#             batched_graph = batched_graph.to(device)
#             text_graph = text_graph.to(device)
#             image_graph = image_graph.to(device)
#
#             sarcasm_logits, _, _ = model(batched_graph, text_graph, image_graph)
#             preds = torch.argmax(sarcasm_logits, dim=1).cpu().numpy()
#             batch_indices = indices.cpu().numpy()
#
#             all_indices.extend(batch_indices)
#             all_preds.extend(preds)
#
#     test_df = pd.read_csv(f'{config.root_dir}{config.me_test_csv_name}', encoding='latin')
#
#     # 转成 DataFrame 并按 index 对齐
#     prediction_df = pd.DataFrame({
#         'index': all_indices,
#         'prediction': all_preds
#     }).sort_values('index').reset_index(drop=True)
#
#     # 假设 test_df 是按 index 顺序来的
#     test_df = test_df.sort_index().reset_index(drop=True)
#     test_df['prediction'] = prediction_df['prediction']
#
#     # 可选：保存结果
#     test_df.to_csv("test_with_predictions.csv", index=False)
#