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

from transformers import get_linear_schedule_with_warmup

import config, Model, dataset, engine_mest, utils

if __name__ == '__main__':
    
    ## Setup the dataset 

    dataset_name = "me" ## me15, we
    utils.set_seed(5)

    if dataset_name == "me":
        dataset_train, dataset_test, dataset_dev = utils.set_up_memotion()
    elif dataset_name == "mu":
        dataset_train, dataset_test, dataset_dev = utils.set_up_mustard()
    else:
        print("No Data")

    ## Setup the dataloaders
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Inititalize the GAME-ON model
    gnn_model = model.MModel()

    ## FInd the number of parameters
    print("Total number of parameters:", sum(p.numel() for p in gnn_model.parameters()))

    gnn_model.to(device)

    ## Calculate number of train steps
    num_update_steps_per_epoch = math.ceil(len(dataloader_train) / config.gradient_accumulation_steps)
    num_train_steps = num_update_steps_per_epoch * config.epochs

    optimizer = AdamW(gnn_model.parameters(), lr=config.lr, weight_decay=1e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.epochs):
        print(f"\n---------------------- Epoch: {epoch + 1} ---------------------------------- \n")
        ## Training Loop

        train_loss, train_report, val_loss, val_report, train_sar_f1, sar_f1 = engine_mest.train_func_epoch(
            epoch + 1, gnn_model, dataloader_train, device, optimizer, scheduler, val_loader=dataloader_dev
        )

        print(f"\nEpoch: {epoch + 1} | Training loss: {train_loss} | Validation Loss: {val_loss}")
        print()
        print("Train Report:")
        print(train_report)
        print("Train Micro F1:")
        print(train_sar_f1)
        print()
        print("Validation Report:")
        print(val_report)
        print("Dev Micro F1:")
        print(sar_f1)
        print()

        if val_loss < best_loss:
            best_val_loss = val_loss
            torch.save(gnn_model.state_dict(), '/home/zxl/MultiTask classification/GAME-ON-main/best_model.pth')
            print(f"Best model saved at epoch {epoch}")

            # Load the best model for evaluation on the test set
        gnn_model.load_state_dict(torch.load('/home/zxl/MultiTask classification/GAME-ON-main/best_model.pth'))
        test_loss, test_report, test_f1 = engine_mest.eval_func(gnn_model, dataloader_test, device)
        print(f"Test Loss: {test_loss}")
        print(f"Test Classification Report: ")
        print(test_report)
        print("Test Micro F1:")
        print(test_f1)
        print()

        print(f"\n----------------------------------------------------------------------------")