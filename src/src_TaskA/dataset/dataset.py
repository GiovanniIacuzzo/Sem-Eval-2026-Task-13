import torch
import pandas as pd
import numpy as np
import os
import re
import logging
import zlib
import math
from collections import Counter
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
            
        codes = dataframe['code'].astype(str).tolist()
        labels = dataframe['label'].astype(int).tolist()
        
        desc = f"[{'Train' if is_train else 'Val'}] Processing & Feature Eng."
        
        for i, code in enumerate(tqdm(codes, desc=desc, leave=False)):
            label = labels[i]
            
            # Estrazione Feature Stilistiche
            stylistic_feats = self._extract_stylistic_features(code)
            
            # Tokenizzazione Ottimizzata
            encoding = self.tokenizer(
                code,
                truncation=True,
                padding=False, 
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            self.input_ids.append(encoding["input_ids"].squeeze(0))
            self.attention_masks.append(encoding["attention_mask"].squeeze(0))
            self.labels.append(torch.tensor(label, dtype=torch.long))
            self.extra_features.append(stylistic_feats)

    def _extract_stylistic_features(self, code):
        """
        Estrae feature statistiche agnostiche rispetto al linguaggio di programmazione.
        Queste feature aiutano quando il modello incontra linguaggi mai visti (OOD).
        """
        features = []
        code_len = len(code) + 1e-9
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # --- 1. Shannon Entropy ---
        counts = Counter(code)
        probs = [c / code_len for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        features.append(entropy / 8.0)

        # --- 2. Zlib Compression Ratio ---
        compressed_len = len(zlib.compress(code.encode('utf-8')))
        features.append(compressed_len / code_len)

        # --- 3. Struttura e Spazi ---
        features.append(code.count(' ') / code_len)
        features.append((code.count('\t') * 4) / code_len)
        
        # --- 4. Densità Simboli "Sintattici" ---
        symbols = len(re.findall(r'[{}()\[\];.,]', code))
        features.append(symbols / code_len)
        
        # --- 5. Statistiche sulle Linee ---
        if non_empty_lines:
            line_lengths = [len(l) for l in non_empty_lines]
            avg_line_len = np.mean(line_lengths)
            std_line_len = np.std(line_lengths)
            
            features.append(min(avg_line_len / 100.0, 1.0)) # Normalizzato
            features.append(min(std_line_len / 50.0, 1.0))  # Normalizzato
        else:
            features.append(0.0)
            features.append(0.0)

        # --- 6. Ratio Commenti/Codice ---
        comment_chars = code.count('#') + code.count('//')
        features.append(comment_chars / code_len)

        # --- 7. Snake vs Camel Case ---
        snake_count = code.count('_')
        camel_count = len(re.findall(r'[a-z][A-Z]', code))
        total_casing = snake_count + camel_count + 1e-9
        features.append(snake_count / total_casing)
        
        # Totale Features: 8
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
    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "validation.parquet")

    logger.info(f"Loading train data from: {train_path}")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # Pulizia base
    train_df = train_df.dropna(subset=['code', 'label'])
    val_df = val_df.dropna(subset=['code', 'label'])
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)

    TARGET_TRAIN = 50000 
    TARGET_VAL = 5000
    
    if len(train_df) > TARGET_TRAIN:
        logger.info(f"Downsampling Train from {len(train_df)} to {TARGET_TRAIN}")
        train_df = train_df.sample(n=TARGET_TRAIN, random_state=42)
        
    if len(val_df) > TARGET_VAL:
        logger.info(f"Downsampling Val from {len(val_df)} to {TARGET_VAL}")
        val_df = val_df.sample(n=TARGET_VAL, random_state=42)
    
    logger.info(f"Final Samples -> Train: {len(train_df)}, Val: {len(val_df)}")

    train_ds = CodeDataset(train_df, tokenizer, max_length=config["max_length"], is_train=True)
    val_ds = CodeDataset(val_df, tokenizer, max_length=config["max_length"], is_train=False)
    
    # Calcolo pesi classi per la Loss
    labels = train_df['label'].values
    classes, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / (len(classes) * counts)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    return train_ds, val_ds, class_weights_tensor