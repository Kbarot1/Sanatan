import pandas as pd
import torch
from transformers import RagTokenizer
from sklearn.preprocessing import LabelEncoder

class HinduScripturesDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, tokenizer, max_len=512):
        self.data = pd.read_csv(data_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(data_file, tokenizer, batch_size=32, max_len=512):
    dataset = HinduScripturesDataset(data_file, tokenizer, max_len)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def create_label_encoder(labels):
    le = LabelEncoder()
    le.fit(labels)
    return le

def encode_labels(labels, le):
    return le.transform(labels)

def decode_labels(labels, le):
    return le.inverse_transform(labels)