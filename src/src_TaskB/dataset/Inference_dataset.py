import pandas as pd
from torch.utils.data import Dataset

class InferenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = str(self.data.at[idx, 'code'])
        label = -1
        if 'label' in self.data.columns:
            val = self.data.at[idx, 'label']
            if pd.notna(val):
                label = int(val)
        
        encoding = self.tokenizer(
            code, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label
        }