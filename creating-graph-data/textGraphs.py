from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.df['text_corrected'][idx]
        unique_id = self.df['image_name'][idx].split(".")[0]

        encoded_sent = self.tokenizer.encode_plus(
            text=text,  # No need for additional preprocessing if tokenizer handles it
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=512,  # Max length to truncate/pad
            padding='max_length',  # Pad sentence to max length
            return_attention_mask=True,  # Return attention mask
            truncation=True
        )

        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'unique_id': unique_id}


def store_data(model, device, df, dataset, store_dir):
    lengths = []
    model.eval()

    for idx in tqdm(range(len(df))):
        sample = dataset.__getitem__(idx)

        input_ids, attention_mask = sample['input_ids'].unsqueeze(0), sample['attention_mask'].unsqueeze(0)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        unique_id = sample['unique_id']
        num_tokens = attention_mask.sum().detach().cpu().item()

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        out_tokens = out.last_hidden_state[:, 1:num_tokens, :].detach().cpu().squeeze(0).numpy()  # token vectors

        filename = f'{root_dir}{store_dir}{unique_id}.npy'
        np.save(filename, out_tokens)

        lengths.append(num_tokens)

        out_cls = out.last_hidden_state[:, 0, :].unsqueeze(0).detach().cpu().squeeze(0).numpy()  # CLS vector
        filename = f'{root_dir}{store_dir}{unique_id}_full_text.npy'
        np.save(filename, out_cls)

    return lengths


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    root_dir = "/home/zxl/MultiTask classification/"

    train_csv_name = "data/train.csv"
    test_csv_name = "data/test.csv"
    dev_csv_name = "data/dev.csv"

    # Loading Qwen2 model and tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model = AutoModel.from_pretrained("Qwen/Qwen2-7B-Instruct").to(device)

    store_dir = "proposed/test2/Large_model/memotion/textfeature/"

    df_train = pd.read_csv(f'{root_dir}{train_csv_name}', encoding='latin')
    train_dataset = TextDataset(df_train, tokenizer)

    lengths = store_data(model, device, df_train, train_dataset, store_dir)

    ## Create graph data for testing set
    df_test = pd.read_csv(f'{root_dir}{test_csv_name}', encoding='latin')
    # df_test = df_test.dropna().reset_index(drop=True)
    test_dataset = TextDataset(df_test, tokenizer)

    lengths = store_data(model, device, df_test, test_dataset, store_dir)

    ## Create graph data for valid set
    df_dev = pd.read_csv(f'{root_dir}{dev_csv_name}', encoding='latin')
    # df_test = df_test.dropna().reset_index(drop=True)
    dev_dataset = TextDataset(df_dev, tokenizer)

    lengths = store_data(model, device, df_dev, dev_dataset, store_dir)