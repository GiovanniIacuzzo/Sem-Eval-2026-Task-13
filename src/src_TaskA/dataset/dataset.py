import torch
import pandas as pd
import numpy as np
import os
import re
import logging
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=1024, is_train=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.extra_features = []
        
        # Standardizzazione nomi colonne
        if 'code' not in dataframe.columns:
            if 'text' in dataframe.columns: 
                dataframe = dataframe.rename(columns={'text': 'code'})
            else:
                raise ValueError(f"Dataset columns must contain 'code'. Found: {dataframe.columns}")
            
        codes = dataframe['code'].astype(str).tolist()
        labels = dataframe['label'].astype(int).tolist()
        
        desc = f"[{'Train' if is_train else 'Val'}] Processing"
        
        for i, code in enumerate(tqdm(codes, desc=desc, leave=False)):
            label = labels[i]
            stylistic_feats = self._extract_stylistic_features(code)
            
            encoding = self.tokenizer(
                code,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            self.input_ids.append(encoding["input_ids"].squeeze(0))
            self.attention_masks.append(encoding["attention_mask"].squeeze(0))
            self.labels.append(torch.tensor(label, dtype=torch.long))
            self.extra_features.append(stylistic_feats)

    def _extract_stylistic_features(self, code):
        """
        Feature extraction manuale (coerente con Task B)
        """
        features = []
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        code_len = len(code) + 1
        
        features.append(code.count(' ') / code_len)
        features.append((code.count('#') + code.count('//')) / code_len)
        features.append(len(re.findall(r'[{}()\[\];.,]', code)) / code_len)
        
        avg_line_len = np.mean([len(l) for l in non_empty_lines]) if non_empty_lines else 0
        features.append(min(avg_line_len / 100.0, 1.0))
        features.append((len(lines) - len(non_empty_lines)) / (len(lines) + 1))
        
        snake_count = code.count('_')
        camel_count = len(re.findall(r'[a-z][A-Z]', code))
        features.append(snake_count / (snake_count + camel_count + 1))
        
        return torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
            "extra_features": self.extra_features[idx] 
        }

def load_data(config, tokenizer):
    data_dir = config["data_dir"]
    
    # Percorsi corretti basati sulla tua struttura file
    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "validation.parquet")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at: {train_path}")

    logger.info(f"Loading train data from: {train_path}")
    train_df = pd.read_parquet(train_path)
    
    logger.info(f"Loading validation data from: {val_path}")
    val_df = pd.read_parquet(val_path)

    train_df = train_df.dropna(subset=['code', 'label'])
    val_df = val_df.dropna(subset=['code', 'label'])
    
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)
    
    logger.info(f"Samples -> Train: {len(train_df)}, Val: {len(val_df)}")

    train_ds = CodeDataset(train_df, tokenizer, max_length=config["max_length"], is_train=True)
    val_ds = CodeDataset(val_df, tokenizer, max_length=config["max_length"], is_train=False)
    
    labels = train_df['label'].values
    classes, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / (len(classes) * counts)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    logger.info(f"Computed Class Weights: {class_weights_tensor}")
    
    return train_ds, val_ds, class_weights_tensor