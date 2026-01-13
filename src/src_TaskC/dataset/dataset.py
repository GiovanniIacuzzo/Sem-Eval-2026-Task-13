import os
import torch
import pandas as pd
import numpy as np
import re
import math
import logging
from collections import Counter
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

logger = logging.getLogger(__name__)

# =============================================================================
# 1. UTILS DI CARICAMENTO & BILANCIAMENTO
# =============================================================================

def get_class_weights(df, device):
    """
    Calcola i pesi delle classi in modo dinamico (funziona sia per 2 che per 4 classi).
    """
    if 'label' not in df.columns:
        raise ValueError("Il dataframe deve contenere la colonna 'label'")
        
    y = df['label'].values
    classes = np.unique(y)
    num_classes = len(classes)
    
    logger.info(f"Computing Class Weights for {num_classes} classes: {classes}")
    
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    
    # Creazione Tensore
    dim = int(np.max(classes) + 1)
    weights_tensor = torch.ones(dim, dtype=torch.float32)
    
    for cls, w in zip(classes, weights):
        weights_tensor[int(cls)] = float(w)
            
    logger.info(f"Class Weights Tensor: {weights_tensor.numpy()}")
    return weights_tensor.to(device)


def load_data_for_training(config):
    """
    Carica i dati grezzi. Il filtraggio o re-mapping delle label 
    viene fatto in train.py (per supportare i vari stage).
    """
    data_dir = config["data"].get("data_dir", "data/Task_C_Processed")
    train_path = os.path.join(data_dir, "train_processed.parquet")
    val_path = os.path.join(data_dir, "val_processed.parquet")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Non trovo i file parquet in {data_dir}.")

    logger.info(f"Loading Training Data from {train_path}...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    max_samples = config["data"].get("max_samples_per_class", None) 
    if max_samples:
        logger.info(f"Applying Class Balancing (Max {max_samples} per class)...")
        balanced_dfs = []
        for label in train_df['label'].unique():
            df_subset = train_df[train_df['label'] == label]
            if len(df_subset) > max_samples:
                df_subset = df_subset.sample(n=max_samples, random_state=42)
            balanced_dfs.append(df_subset)
        train_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    return train_df, val_df

# =============================================================================
# 2. DATASET CLASS ROBUSTA
# =============================================================================

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, is_train=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.extra_features = []
        
        dataframe = dataframe.dropna(subset=['code', 'label']).copy()
        
        codes = dataframe['code'].astype(str).tolist()
        raw_labels = dataframe['label'].astype(int).tolist()
        
        desc = f"[{'Train' if is_train else 'Val'}] Processing"
        
        for i, code in enumerate(tqdm(codes, desc=desc, dynamic_ncols=True)):
            label = raw_labels[i]
            
            stylistic_feats = self._extract_robust_features(code)
            
            tokens = tokenizer.tokenize(code)
            capacity = max_length - 2
            
            processed_tokens = []
            
            if len(tokens) <= capacity:
                processed_tokens = tokens
            else:
                if self.is_train:
                    start_idx = np.random.randint(0, len(tokens) - capacity)
                    processed_tokens = tokens[start_idx : start_idx + capacity]
                else:
                    processed_tokens = tokens[:capacity]

            chunk_str = tokenizer.convert_tokens_to_string(processed_tokens)

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

    def _extract_robust_features(self, code):
        """
        Sostituisce le feature 'fragili' del Task B con metriche resistenti all'offuscamento.
        Output: Tensore (8,)
        """
        features = []
        code_len = len(code) + 1
        
        counts = Counter(code)
        entropy = -sum((cnt / code_len) * math.log2(cnt / code_len) for cnt in counts.values())
        features.append(entropy / 8.0)
        
        specials = len(re.findall(r'[{}()\[\];.,+\-*/%&|^!=<>?]', code))
        features.append(specials / code_len)
        
        features.append(code.count(' ') / code_len)
        
        words = re.findall(r'\w+', code)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        features.append(min(avg_word_len / 20.0, 1.0))

        keywords = len(re.findall(r'\b(if|for|while|return|def|class|import|void|int)\b', code))
        features.append(keywords / (len(words) + 1))
        
        unique_ratio = len(set(words)) / len(words) if words else 0
        features.append(unique_ratio)
        
        long_strings = len(re.findall(r'"[^"]{50,}"|\'[^\']{50,}\'', code))
        features.append(min(long_strings / 5.0, 1.0))
        
        current_depth = 0
        max_depth = 0
        for char in code:
            if char == '{': current_depth += 1
            elif char == '}': current_depth = max(0, current_depth - 1)
            max_depth = max(max_depth, current_depth)
        features.append(min(max_depth / 10.0, 1.0))

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