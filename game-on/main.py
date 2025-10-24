# Importing libraries
import numpy as np
import math
from dgl.dataloading import GraphDataLoader
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import config, Model, engine_test, utils


if __name__ == '__main__':
    # Setup the dataset
    dataset_name = "me"  ## me, mu
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inititalize the model
    gnn_model = Model.MModel()

    # FInd the number of parameters
    print("Total number of parameters:", sum(p.numel() for p in gnn_model.parameters()))

    gnn_model.to(device)

    # Calculate number of train steps
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

        train_loss, report_sar, report_sent, val_loss, val_report_sar, val_report_sent = (
            engine_test.train_func_epoch(epoch + 1, gnn_model, dataloader_train, device, optimizer, scheduler,
                                    val_loader=dataloader_dev))

        print(f"\nEpoch: {epoch + 1} | Training loss: {train_loss} | Validation Loss: {val_loss}")
        print()
        print("Train Sarcasm:")
        print(f"\n- Precision: {report_sar['weighted avg']['precision']: .4f} || -Recall: {report_sar['weighted avg']['recall']: .4f} || -F1: {report_sar['weighted avg']['f1-score']: .4f}")

        print("Train Sentiment:")
        print(f"\n- Precision: {report_sent['weighted avg']['precision']: .4f} || -Recall: {report_sent['weighted avg']['recall']: .4f} || -F1: {report_sent['weighted avg']['f1-score']: .4f}")
        # print(
        #     f"\nTrain Sar Micro F1: {train_sar_f1} | Train Sent Micro F1: {trai_sent_f1}")
        print()
        print("Validation Sar Report:")
        print(f"\n- Precision: {val_report_sar['weighted avg']['precision']: .4f} || -Recall: {val_report_sar['weighted avg']['recall']: .4f} || -F1: {val_report_sar['weighted avg']['f1-score']: .4f}")
        print("Validation Sent Report:")
        print(
            f"\n- Precision: {val_report_sent['weighted avg']['precision']: .4f} || -Recall: {val_report_sent['weighted avg']['recall']: .4f} || -F1: {val_report_sent['weighted avg']['f1-score']: .4f}")
        print()

        if val_loss < best_loss:
            best_val_loss = val_loss
            torch.save(gnn_model.state_dict(), '/home/zxl/MultiTask classification/proposed/test2/memotion/best_model1.pth')
            print(f"Best model saved at epoch {epoch}")

            # Load the best model for evaluation on the test set
        gnn_model.load_state_dict(torch.load('/home/zxl/MultiTask classification/proposed/test2/memotion/best_model1.pth'))
        test_loss, test_report_sar, test_report_sent = engine_test.eval_func(
            gnn_model, dataloader_test, device)
        print(f"Test Loss: {test_loss}")
        print()
        print("Test Sar Report:")
        print(f"\n- Precision: {test_report_sar['weighted avg']['precision']: .4f} || -Recall: {test_report_sar['weighted avg']['recall']: .4f} || -F1: {test_report_sar['weighted avg']['f1-score']: .4f}")
        print("Test Sent Report:")
        print(f"\n- Precision: {test_report_sent['weighted avg']['precision']: .4f} || -Recall: {test_report_sent['weighted avg']['recall']: .4f} || -F1: {test_report_sent['weighted avg']['f1-score']: .4f}")
        print()


        print(f"\n----------------------------------------------------------------------------")
