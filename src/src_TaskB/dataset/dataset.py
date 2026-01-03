import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

LANG_MAP = {
    "python": 0, "java": 1, "cpp": 2, "c": 3, "c#": 4, "cs": 4, 
    "javascript": 5, "php": 6, "ruby": 7, "rust": 8, "go": 9,
    "typescript": 10, "kotlin": 11, "swift": 12, "scala": 13, "shell": 14
}

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, mode="binary", is_train=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.is_train = is_train
        
        self.samples = []
        
        codes = dataframe['code'].astype(str).tolist()
        
        if self.mode == "binary":
            labels = dataframe['is_ai'].astype(int).tolist()
        elif self.mode == "families":
            labels = dataframe['family_label'].astype(int).tolist()
        
        stride = 256
        
        print(f"[{'Train' if is_train else 'Val'}] Processing {len(codes)} files...")
        
        for i, code in enumerate(codes):
            label = labels[i]
            
            tokens = tokenizer.tokenize(code)
            
            capacity = max_length - 2
            
            if len(tokens) <= capacity:
                self.samples.append({
                    "code_str": code,
                    "label": label
                })
            else:
                if is_train:
                    for start_idx in range(0, len(tokens), stride):
                        end_idx = min(start_idx + capacity, len(tokens))
                        chunk_tokens = tokens[start_idx:end_idx]
                        
                        chunk_str = tokenizer.convert_tokens_to_string(chunk_tokens)
                        
                        self.samples.append({
                            "code_str": chunk_str,
                            "label": label
                        })
                        
                        if end_idx >= len(tokens):
                            break
                else:
                    half = capacity // 2
                    head = tokens[:half]
                    tail = tokens[-half:]
                    chunk_str = tokenizer.convert_tokens_to_string(head + tail)
                    self.samples.append({
                        "code_str": chunk_str,
                        "label": label
                    })

        print(f"[{'Train' if is_train else 'Val'}] Total Samples generated: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        code = sample["code_str"]
        label = sample["label"]

        encoding = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(-1, dtype=torch.long), 
            "extra_features": torch.tensor([], dtype=torch.float) 
        }

def load_data(config, tokenizer, mode="binary"):
    common_cfg = config["data"]
    data_dir = common_cfg.get("data_dir", "data/Task_B_Processed")
    
    train_file = f"train_{mode}.parquet"
    val_file = f"val_{mode}.parquet"
    
    train_path = os.path.join(data_dir, train_file)
    val_path = os.path.join(data_dir, val_file)
    
    print(f"Loading Parquet files from {data_dir}...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    class_weights_tensor = None
    if mode == "families":
        print("Computing Class Weights (Balanced)...")
        y_train = train_df['family_label'].values
        classes = np.unique(y_train)
        cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights_tensor = torch.tensor(cw, dtype=torch.float)

    train_ds = CodeDataset(
        train_df, 
        tokenizer, 
        max_length=common_cfg["max_length"], 
        mode=mode, 
        is_train=True
    )

    val_ds = CodeDataset(
        val_df, 
        tokenizer, 
        max_length=common_cfg["max_length"], 
        mode=mode, 
        is_train=False
    )

    return train_ds, val_ds, class_weights_tensor