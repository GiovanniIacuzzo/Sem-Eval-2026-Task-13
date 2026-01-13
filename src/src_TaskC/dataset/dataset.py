import os
import torch
import pandas as pd
import numpy as np
import re
import logging
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

logger = logging.getLogger(__name__)

# =============================================================================
# 1. UTILS DI CARICAMENTO & BILANCIAMENTO
# =============================================================================

def get_class_weights(df, device):
    """
    Calcola i pesi delle classi inversamente proporzionali alla loro frequenza.
    Fondamentale per la Focal Loss nel Task C dove le classi 2 e 3 sono rare.
    """
    if 'label' not in df.columns:
        raise ValueError("Il dataframe deve contenere la colonna 'label'")
        
    y = df['label'].values
    classes = np.unique(y)
    
    if len(classes) < 4:
        logger.warning(f"Attenzione: Trovate solo {len(classes)} classi nel training set: {classes}")
    
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    
    weights_tensor = torch.ones(4, dtype=torch.float32)
    for cls, w in zip(classes, weights):
        if cls < 4:
            weights_tensor[int(cls)] = float(w)
            
    logger.info(f"Class Weights Computed: {weights_tensor.numpy()}")
    return weights_tensor.to(device)


def load_data_for_training(config):
    """
    Carica i dati di training e validation.
    Applica opzionalmente strategie di downsampling per le classi maggioritarie.
    """
    data_dir = config["data"].get("data_dir", "data/Task_C_Processed")
    train_path = os.path.join(data_dir, "train_processed.parquet")
    val_path = os.path.join(data_dir, "val_processed.parquet")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Non trovo i file parquet in {data_dir}. Hai eseguito prepare_data.py?")

    logger.info(f"Loading Training Data from {train_path}...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    max_samples = config["data"].get("max_samples_per_class", None) 
    
    if max_samples:
        logger.info(f"Applying Class Balancing (Max samples per class: {max_samples})...")
        balanced_dfs = []
        for label in sorted(train_df['label'].unique()):
            df_subset = train_df[train_df['label'] == label]
            if len(df_subset) > max_samples:
                df_subset = df_subset.sample(n=max_samples, random_state=42)
            balanced_dfs.append(df_subset)
        
        train_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"New Training Size: {len(train_df)}")
        logger.info(f"New Distribution:\n{train_df['label'].value_counts()}")

    return train_df, val_df

# =============================================================================
# 2. DATASET CLASS
# =============================================================================

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, is_train=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.extra_features = []
        
        dataframe = dataframe.dropna(subset=['code', 'label']).copy()
        
        codes = dataframe['code'].astype(str).tolist()
        raw_labels = dataframe['label'].astype(int).tolist()
        
        desc = f"[{'Train' if is_train else 'Val'}] Processing"
        
        # Loop principale
        for i, code in enumerate(tqdm(codes, desc=desc, dynamic_ncols=True)):
            label = raw_labels[i]
            
            stylistic_feats = self._extract_stylistic_features(code)
            
            tokens = tokenizer.tokenize(code)
            capacity = max_length - 2
            
            if len(tokens) <= capacity:
                chunk_str = code
            else:
                half = capacity // 2
                kept_tokens = tokens[:half] + tokens[-half:]
                chunk_str = tokenizer.convert_tokens_to_string(kept_tokens)

            encoding = self.tokenizer(
                chunk_str,
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
        L'estrattore del Task B che ha funzionato bene.
        Include metriche vitali per rilevare Adversarial Code.
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
        
        logic_tokens = len(re.findall(r'\b(if|for|while|return|switch|case|break)\b', code))
        features.append(logic_tokens / (len(code.split()) + 1))
        
        max_indent = 0
        if non_empty_lines:
            max_indent = max([len(l) - len(l.lstrip()) for l in non_empty_lines])
        features.append(min(max_indent / 20.0, 1.0))
        
        return torch.tensor(features, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
            "extra_features": self.extra_features[idx] 
        }