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

logger = logging.getLogger(__name__)

# =============================================================================
# 1. UTILS DI CARICAMENTO & BILANCIAMENTO
# =============================================================================

def get_class_weights(df, device):
    """
    Calcola i pesi delle classi in modo dinamico.
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
    data_dir = config["data"].get("data_dir", "data/Task_C_Processed")
    
    train_path = os.path.join(data_dir, "featured_train_processed.parquet")
    val_path = os.path.join(data_dir, "featured_val_processed.parquet")
    
    if not os.path.exists(train_path):
        logger.warning(f"File ottimizzato {train_path} non trovato. Provo quello standard...")
        train_path = os.path.join(data_dir, "train_processed.parquet")
    else:
        logger.info(f"Trovato dataset ottimizzato: {train_path}")

    if not os.path.exists(val_path):
        val_path = os.path.join(data_dir, "val_processed.parquet")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Non trovo file parquet in {data_dir}")

    logger.info(f"Loading Training Data from {train_path}...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # Opzionale: Balancing subsampling
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
# 2. DATASET LAZY LOADING
# =============================================================================
class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, is_train=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        # Filtriamo subito eventuali righe corrotte
        dataframe = dataframe.dropna(subset=['code', 'label']).copy()
        
        # 1. Carichiamo Codice e Label
        self.codes = dataframe['code'].astype(str).tolist()
        self.labels = dataframe['label'].astype(int).tolist()
        
        # 2. Gestione Feature Pre-calcolate
        if 'extra_features' in dataframe.columns:
            logger.info("⚡ Utilizzo features pre-calcolate")
            self.extra_features = dataframe['extra_features'].tolist()
        else:
            logger.warning(" Features pre-calcolate NON trovate. Calcolo on-the-fly attivo.")
            self.extra_features = None
        
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # A. Recupero Feature Stilistiche
        if self.extra_features is not None:
            feats_data = self.extra_features[idx]
            if isinstance(feats_data, np.ndarray):
                feats_data = feats_data.tolist()
            extra_features = torch.tensor(feats_data, dtype=torch.float)
        else:
            extra_features = self._extract_robust_features(self.codes[idx])

        # B. Tokenizzazione
        code = self.codes[idx]
        label = self.labels[idx]

        all_ids = self.tokenizer.encode(code, add_special_tokens=True, truncation=False)
        total_len = len(all_ids)
        
        if total_len > self.max_length:
            if self.is_train:
                # Random Crop: [CLS] + chunk casuale
                start_idx = np.random.randint(1, total_len - self.max_length + 1)
                input_ids = [all_ids[0]] + all_ids[start_idx : start_idx + self.max_length - 1]
            else:
                # Truncation standard
                input_ids = all_ids[:self.max_length]
        else:
            input_ids = all_ids

        # C. Padding
        processed_len = len(input_ids)
        padding_len = self.max_length - processed_len
        
        if padding_len > 0:
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = [1] * processed_len + [0] * padding_len
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = [1] * self.max_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
            "extra_features": extra_features
        }

    def _extract_robust_features(self, code):
        """
        Fallback per calcolo on-the-fly (usato se non ci sono features pre-calcolate o in inferenza).
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
        num_words = len(words)
        if num_words > 0:
            avg_word_len = sum(len(w) for w in words) / num_words
            unique_ratio = len(set(words)) / num_words
        else:
            avg_word_len = 0
            unique_ratio = 0
            
        features.append(min(avg_word_len / 20.0, 1.0))
        features.append(len(re.findall(r'\b(if|for|while|return|def|class|import|void|int)\b', code)) / (num_words + 1))
        features.append(unique_ratio)
        features.append(min(len(re.findall(r'"[^"]{50,}"|\'[^\']{50,}\'', code)) / 5.0, 1.0))
        
        current_depth = 0
        max_depth = 0
        for char in code:
            if char == '{': 
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}': 
                current_depth = max(0, current_depth - 1)
        features.append(min(max_depth / 10.0, 1.0))

        return torch.tensor(features, dtype=torch.float)