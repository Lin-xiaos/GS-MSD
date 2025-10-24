# Importing libraries
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report,  f1_score
import config


def train_func_epoch(epoch, model, data_loader, device, optimizer, scheduler, val_loader=None, gamma_1=0.5, gamma_2=0.5, gamma_3=1.0):
    """Function for a single training epoch

    Args:
        epoch (int): current epoch number
        model (nn.Module)
        data_loader (GraphDataLoader)
        device (str)
        optimizer (torch.optim)
        scheduler (torch.optim.lr_scheduler)
        val_loader (GraphDataLoader, optional): Validation data loader. Defaults to None.
    """

    model.train()
    total_loss = 0
    targets = []
    predictions = []
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    with tqdm(data_loader, unit="batch", total=len(data_loader)) as single_epoch:
        outputsar, outputsent, outputemo = [], [], []
        tarsar, tarsent, taremo = [], [], []
        for step, batch in enumerate(single_epoch):
            single_epoch.set_description(f"Training- Epoch {epoch}")

            batched_graph, text_graph, image_graph, sarcasm, sentiment = batch
            batched_graph, text_graph, image_graph, sarcasm, sentiment = (
                batched_graph.to(device),
                text_graph.to(device),
                image_graph.to(device),
                sarcasm.to(device),
                sentiment.to(device)
            )

            sar, sent, js_loss = model(batched_graph, text_graph, image_graph)

            label_sar = sarcasm.cpu().detach().numpy()
            label_sent = sentiment.cpu().detach().numpy()
            pred_sar = np.argmax(
                sar.cpu().detach().numpy(), axis=1)
            pred_sent = np.argmax(
                sent.cpu().detach().numpy(), axis=1)
            outputsar.append(pred_sar)
            outputsent.append(pred_sent)
            tarsar.append(label_sar)
            tarsent.append(label_sent)

            outputsar.append(pred_sar)
            outputsent.append(pred_sent)
            tarsar.append(label_sar)
            tarsent.append(label_sent)

            loss1 = criterion(sar, sarcasm)
            loss2 = criterion(sent, sentiment)

            loss = gamma_1 * loss1 + gamma_2 * loss2 + gamma_3 * js_loss
            # loss.requires_grad_(True)
            total_loss += loss.item()

            loss.backward()

            if step % config.gradient_accumulation_steps == 0 or step == len(data_loader) - 1:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            single_epoch.set_postfix(train_loss=total_loss / (step + 1))

    outputsar = np.concatenate(outputsar, axis=0)
    outputsent = np.concatenate(outputsent, axis=0)
    tarsar = np.concatenate(tarsar, axis=0)
    tarsent = np.concatenate(tarsent, axis=0)

    print(tarsar)
    print(outputsar)

    train_sar_f1 = f1_score(tarsar, outputsar, average='micro')

    train_sent_f1 = f1_score(tarsent, outputsent, average='micro')

    report_sar = classification_report(tarsar, outputsar, output_dict=True, labels=np.unique(tarsar))
    report_sent = classification_report(tarsent, outputsent, output_dict=True, labels=np.unique(tarsent))

    epoch_train_loss = total_loss / len(data_loader)

    # Run validation if val_loader is provided
    if val_loader:
        val_loss, val_report_sar, val_report_sent = eval_func(model, val_loader,  device, epoch)
        return epoch_train_loss, report_sar, report_sent, val_loss, val_report_sar, val_report_sent

    return epoch_train_loss, report_sar


def eval_func(model, data_loader, device, epoch=1, gamma_1=0.5, gamma_2=0.5, gamma_3=1.0):
    """Function for a single validation epoch

    Args:
        epoch (int): current epoch number
        model (nn.Module)
        data_loader (GraphDataLoader)
        device (str)
    """

    model.eval()
    total_loss = 0
    targets = []
    predictions = []
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).cuda()

    with tqdm(data_loader, unit="batch", total=len(data_loader)) as single_epoch:
        outputsar, outputsent, outputemo = [], [], []
        tarsar, tarsent, taremo = [], [], []
        for step, batch in enumerate(single_epoch):
            single_epoch.set_description(f"Evaluating- Epoch {epoch}")

            batched_graph, text_graph, image_graph, sarcasm, sentiment = batch
            batched_graph, text_graph, image_graph, sarcasm, sentiment = (
                batched_graph.to(device),
                text_graph.to(device),
                image_graph.to(device),
                sarcasm.to(device),
                sentiment.to(device)
            )

            with torch.no_grad():
                sar, sent, js_loss = model(batched_graph, text_graph, image_graph)

            loss1_val = criterion(sar, sarcasm)
            loss2_val = criterion(sent, sentiment)
            loss_val = gamma_1 * loss1_val + gamma_2 * loss2_val + gamma_3 * js_loss
            total_loss += loss_val.item()
            single_epoch.set_postfix(loss=loss_val.item())

            label_sar = sarcasm.cpu().detach().numpy()
            label_sent = sentiment.cpu().detach().numpy()

            pred_sar = np.argmax(
                sar.cpu().detach().numpy(), axis=1)
            pred_sent = np.argmax(
                sent.cpu().detach().numpy(), axis=1)

            outputsar.append(pred_sar)
            outputsent.append(pred_sent)
            tarsar.append(label_sar)
            tarsent.append(label_sent)

        outputsar = np.concatenate(outputsar, axis=0)
        outputsent = np.concatenate(outputsent, axis=0)
        tarsar = np.concatenate(tarsar, axis=0)
        tarsent = np.concatenate(tarsent, axis=0)

    epoch_validation_loss = total_loss / len(data_loader)

    report_sar = classification_report(tarsar, outputsar, output_dict=True, labels=np.unique(tarsar))
    report_sent = classification_report(tarsent, outputsent, output_dict=True, labels=np.unique(tarsent))

    return epoch_validation_loss, report_sar, report_sent
