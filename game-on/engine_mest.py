# Importing libraries
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import config


def train_func_epoch(epoch, model, data_loader, device, optimizer, scheduler, val_loader=None):
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
        for step, batch in enumerate(single_epoch):
            single_epoch.set_description(f"Training- Epoch {epoch}")

            batched_graph, text_graph, image_graph, sarcasm, sentiment, emotion = batch
            batched_graph, text_graph, image_graph, sarcasm, sentiment, emotion = (
                batched_graph.to(device),
                text_graph.to(device),
                image_graph.to(device),
                sarcasm.to(device),
                sentiment.to(device),
                emotion.to(device)
            )

            batch_logits = model(batched_graph, text_graph, image_graph)

            label_sar = sarcasm.cpu().detach().numpy()
            label_sent = sentiment.cpu().detach().numpy()
            label_emo = emotion.cpu().detach().numpy()
            pred_multimodal = torch.argmax(batch_logits, dim=1).flatten().cpu().numpy()

            predictions.append(pred_multimodal)
            targets.append(label_emo)

            loss = criterion(batch_logits, emotion)
            total_loss += loss.item()

            loss.backward()

            if step % config.gradient_accumulation_steps == 0 or step == len(data_loader) - 1:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            single_epoch.set_postfix(train_loss=total_loss / (step + 1))

    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)


    train_sar_f1 = f1_score(targets, predictions, average='micro')

    # train_sent_f1 = f1_score(targets, predictions, average='micro')

    # train_emo_f1 = f1_score(targets, predictions, average='micro')

    report = classification_report(targets, predictions, output_dict=True, labels=np.unique(targets))
    epoch_train_loss = total_loss / len(data_loader)

    # Run validation if val_loader is provided
    if val_loader:
        val_loss, val_report, sar_f1 = eval_func(model, val_loader, device, epoch)
        return epoch_train_loss, report, val_loss, val_report, train_sar_f1, sar_f1

    return epoch_train_loss, report


def eval_func(model, data_loader, device, epoch=1):
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
        for step, batch in enumerate(single_epoch):
            single_epoch.set_description(f"Evaluating- Epoch {epoch}")

            batched_graph, text_graph, image_graph, sarcasm, sentiment, emotion = batch
            batched_graph, text_graph, image_graph, sarcasm, sentiment, emotion = (
                batched_graph.to(device),
                text_graph.to(device),
                image_graph.to(device),
                sarcasm.to(device),
                sentiment.to(device),
                emotion.to(device)
            )

            with torch.no_grad():
                batch_logits = model(batched_graph, text_graph, image_graph)

            loss = criterion(batch_logits, emotion)
            total_loss += loss.item()
            single_epoch.set_postfix(loss=loss.item())

            label_sar = sarcasm.cpu().detach().numpy()
            label_sent = sentiment.cpu().detach().numpy()
            label_emo = emotion.cpu().detach().numpy()
            pred_multimodal = torch.argmax(batch_logits, dim=1).flatten().cpu().numpy()
            predictions.append(pred_multimodal)
            targets.append(label_emo)

    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    epoch_validation_loss = total_loss / len(data_loader)

    sar_f1 = f1_score(targets, predictions, average='micro')

    # sent_f1 = f1_score(targets, predictions, average='micro')

    # emo_f1 = f1_score(targets, predictions, average='micro')

    report = classification_report(targets, predictions, output_dict=True, labels=np.unique(targets))

    return epoch_validation_loss, report, sar_f1