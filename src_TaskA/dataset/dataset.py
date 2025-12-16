import os
import random
import logging
import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Logger Setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# -----------------------------------------------------------------------------
# PyTorch Dataset Class
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
    """
    Optimized PyTorch Dataset for T4 GPU.
    Handles tokenization on-the-fly to save RAM.
    """
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 tokenizer, 
                 language_map: Dict[str, int],
                 max_length: int = 512, # Aumentato per T4 (regge fino a 512 comodo)
                 augment: bool = False):
        
        # Reset index is crucial for fast __getitem__ access
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.language_map = language_map
        self.max_length = max_length
        self.augment = augment
        self.mask_token_id = tokenizer.mask_token_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Fast access using .at avoids overhead of .loc
        code = str(self.data.at[idx, "code"])
        label = int(self.data.at[idx, "label"])
        lang_str = str(self.data.at[idx, "language"]).lower()
        
        # DANN Target: Map string to ID. Return -1 if language is unknown
        lang_id = self.language_map.get(lang_str, -1)
        
        # ---------------------------------------------------------
        # 1. Random Cropping Strategy (Generalization)
        # ---------------------------------------------------------
        # Se il codice è molto lungo, prendiamo un pezzo casuale invece che solo l'inizio.
        # Questo aiuta il modello a vedere parti diverse del codice (import vs logica).
        if self.augment and len(code) > self.max_length * 4:
            start_idx = random.randint(0, len(code) - self.max_length * 4)
            # Prendiamo una finestra leggermente più grande del max_length caratteri
            code = code[start_idx : start_idx + self.max_length * 6]

        # ---------------------------------------------------------
        # 2. Tokenization
        # ---------------------------------------------------------
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # ---------------------------------------------------------
        # 3. Smart Augmentation (Token-Level Masking)
        # ---------------------------------------------------------
        # Eseguito solo su Tensor GPU/CPU veloce, non su stringhe
        if self.augment:
            # 15% probability of masking a token
            probability_matrix = torch.full(input_ids.shape, 0.15)
            
            # Don't mask special tokens ([CLS], [SEP], [PAD])
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                input_ids.tolist(), already_has_special_tokens=True
            )
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            
            # Don't mask padding
            probability_matrix.masked_fill_(attention_mask == 0, value=0.0)
            
            masked_indices = torch.bernoulli(probability_matrix).bool()
            input_ids[masked_indices] = self.mask_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long)
        }

# -----------------------------------------------------------------------------
# Data Preprocessing Logic
# -----------------------------------------------------------------------------
def load_and_preprocess(file_path: str, max_code_chars: int = 20000) -> pd.DataFrame:
    """Caricamento veloce con filtri base."""
    columns = ['code', 'label', 'language']
    try:
        df = pd.read_parquet(file_path, columns=columns)
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise e
    
    # Pulizia rapida
    df['language'] = df['language'].str.lower()
    df = df.dropna(subset=['code']).reset_index(drop=True)
    
    # Filtra snippet troppo corti (rumore) o vuoti
    df = df[df['code'].str.len() > 20].copy()
    
    # Troncamento caratteri per risparmiare RAM prima della tokenizzazione
    df['code'] = df['code'].str.slice(0, max_code_chars)
    
    return df

def balance_languages(df: pd.DataFrame, target_samples: int = 3000) -> pd.DataFrame:
    """
    DOWNSAMPLING INTELLIGENTE:
    1. Identifica i linguaggi.
    2. Se un linguaggio ha > target_samples (es. Python con 20k righe), lo taglia a target_samples.
    3. Se ne ha meno, li tiene tutti (o fa upsampling se volessimo, ma meglio evitare duplicati qui).
    """
    logger.info(f"Balancing languages... Target cap per language: {target_samples}")
    
    df_list = []
    # Conta le occorrenze
    lang_counts = df['language'].value_counts()
    logger.info(f"Original distribution:\n{lang_counts.head()}")

    for lang, count in lang_counts.items():
        lang_group = df[df['language'] == lang]
        
        if count > target_samples:
            # Downsampling: prendi un campione casuale
            df_list.append(lang_group.sample(n=target_samples, random_state=42))
        else:
            # Keep all
            df_list.append(lang_group)
            
    balanced_df = pd.concat(df_list).sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Balanced distribution:\n{balanced_df['language'].value_counts().head()}")
    return balanced_df

# -----------------------------------------------------------------------------
# Main Data Loading Interface
# -----------------------------------------------------------------------------
def load_data(config: dict, tokenizer) -> Tuple[CodeDataset, CodeDataset, pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completa di caricamento dati.
    """
    logger.info(">>> Loading Data for Task A <<<")
    
    train_path = config["data"]["train_path"]
    val_path = config["data"]["val_path"]
    
    # 1. Load DataFrames
    train_df = load_and_preprocess(train_path)
    val_df   = load_and_preprocess(val_path)
    
    # 2. Downsampling del linguaggio dominante
    # Leggiamo il parametro dal config, default 4000 (buono per T4/CodeBERT)
    samples_cap = config["data"].get("samples_per_lang", 4000)
    
    if config["data"].get("balance_languages", True):
        train_df = balance_languages(train_df, target_samples=samples_cap)
        # Non bilanciamo il validation set! Deve riflettere la distribuzione reale (o quella del test).
    
    # 3. DANN Language Mapping setup
    # Mappiamo i linguaggi presenti nel config a interi.
    target_langs = config["model"].get("languages", ["python", "java", "cpp", "c#", "javascript"])
    language_map = {lang.lower(): i for i, lang in enumerate(target_langs)}
    
    # 4. Dataset Creation
    max_len = config["data"].get("max_length", 512)
    
    # Train: Augmentation ON
    train_dataset = CodeDataset(
        train_df, 
        tokenizer, 
        language_map, 
        max_length=max_len, 
        augment=True
    )
    
    # Val: Augmentation OFF
    val_dataset = CodeDataset(
        val_df, 
        tokenizer, 
        language_map, 
        max_length=max_len, 
        augment=False
    )

    logger.info(f"Datasets Ready. Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    return train_dataset, val_dataset, train_df, val_df

# -----------------------------------------------------------------------------
# Unit Testing
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import AutoTokenizer
    # Simple test to verify shapes
    try:
        print("Testing Dataset...")
        tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        data = {
            'code': ["print('hello')" * 100, "int x=0;", "def foo(): pass"],
            'label': [0, 1, 0],
            'language': ['python', 'cpp', 'python'] # Python dominante
        }
        df = pd.DataFrame(data)
        
        # Test Balancing
        # Se target=1, python dovrebbe scendere da 2 a 1 campione
        balanced = balance_languages(df, target_samples=1)
        assert len(balanced) == 2 # 1 python + 1 cpp
        print("Balancing Logic OK.")
        
        # Test Dataset Item
        ds = CodeDataset(balanced, tok, {'python': 0, 'cpp': 1}, max_length=128, augment=True)
        item = ds[0]
        print("Item keys:", item.keys())
        print("Input shape:", item['input_ids'].shape)
        print("Test Passed.")
    except Exception as e:
        print(f"Test Failed: {e}")