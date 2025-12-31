import os
import random
import logging
import pandas as pd
import torch
import numpy as np
from typing import Dict
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

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
# Dataset Class
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
    """
    Dataset ottimizzato per Task C.
    Supporta augmentation strutturale (spazi/newlines) per il training.
    """
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 tokenizer, 
                 language_map: Dict[str, int],
                 max_length: int = 512, 
                 augment: bool = False):
        
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.language_map = language_map
        self.max_length = max_length
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def _structural_noise(self, code: str) -> str:
        """
        Simula variazioni di stile umano (spaziature, righe vuote)
        senza distruggere la sintassi.
        """
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            if random.random() < 0.05: new_lines.append("")
            if random.random() < 0.05: line = line + " " * random.randint(1, 3)
            if random.random() < 0.01: line = " " + line
            new_lines.append(line)
        return "\n".join(new_lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        code = str(row["code"])
        label = int(row["label"]) 
        lang_str = str(row["language"]).lower()
        lang_id = self.language_map.get(lang_str, -1)
        
        if self.augment:
            code = self._structural_noise(code)

        tokens = self.tokenizer.tokenize(code)
        
        max_tokens = self.max_length - 2
        
        if len(tokens) > max_tokens:
            half_len = max_tokens // 2
            tokens = tokens[:half_len] + tokens[-half_len:]
        
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long)
        }

# -----------------------------------------------------------------------------
# Data Helpers
# -----------------------------------------------------------------------------
def load_and_preprocess(file_path: str, max_code_chars: int = 20000) -> pd.DataFrame:
    columns = ['code', 'label', 'language'] 
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        df = pd.read_parquet(file_path, columns=columns)
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise e
        
    df['language'] = df['language'].str.lower()
    df = df.dropna(subset=['code']).reset_index(drop=True)
    df = df[df['code'].str.len() > 15].copy()
    df['code'] = df['code'].str.slice(0, max_code_chars)
    return df

def balance_languages(df: pd.DataFrame, target_samples: int = 3000) -> pd.DataFrame:
    """Cap a sample per language to avoid Python dominance."""
    logger.info(f"Balancing languages... Max cap: {target_samples}")
    df_list = []
    for lang, count in df['language'].value_counts().items():
        group = df[df['language'] == lang]
        if count > target_samples:
            df_list.append(group.sample(n=target_samples, random_state=42))
        else:
            df_list.append(group)
    return pd.concat(df_list).sample(frac=1, random_state=42).reset_index(drop=True)

def get_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Inverse frequency weights for Focal Loss."""
    labels = df['label'].values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def get_dynamic_language_map(df: pd.DataFrame) -> Dict[str, int]:
    """Crea la mappa lingue basata sui dati reali presenti."""
    unique_langs = sorted(df['language'].unique())
    lang_map = {lang: i for i, lang in enumerate(unique_langs)}
    logger.info(f"Dynamic DANN Language Map ({len(lang_map)}): {lang_map}")
    return lang_map

# -----------------------------------------------------------------------------
# K-Fold Hook
# -----------------------------------------------------------------------------
def load_data_for_kfold(config: dict) -> pd.DataFrame:
    """
    Carica e unisce Train+Val in un unico DataFrame pulito.
    NON crea ancora i dataset PyTorch (quello lo farà il main loop).
    """
    logger.info(">>> Loading Raw Data for K-Fold Merge <<<")
    
    train_path = config["data"]["train_path"]
    val_path = config["data"]["val_path"]
    
    df_train = load_and_preprocess(train_path)
    df_val = load_and_preprocess(val_path)
    
    full_df = pd.concat([df_train, df_val]).reset_index(drop=True)
    
    samples_cap = config["data"].get("samples_per_lang", 4000)
    if config["data"].get("balance_languages", True):
        full_df = balance_languages(full_df, target_samples=samples_cap)
        
    logger.info(f"Total Merged Data: {len(full_df)} samples")
    return full_df

# -----------------------------------------------------------------------------
# Legacy Loader
# -----------------------------------------------------------------------------
def load_data(config: dict, tokenizer, device=torch.device("cpu")):
    full_df = load_data_for_kfold(config)
    lang_map = get_dynamic_language_map(full_df)
    weights = get_class_weights(full_df, device)
    
    mask = np.random.rand(len(full_df)) < 0.8
    train_df = full_df[mask]
    val_df = full_df[~mask]
    
    train_ds = CodeDataset(train_df, tokenizer, lang_map, config["data"]["max_length"], augment=True)
    val_ds = CodeDataset(val_df, tokenizer, lang_map, config["data"]["max_length"], augment=False)
    
    return train_ds, val_ds, weights