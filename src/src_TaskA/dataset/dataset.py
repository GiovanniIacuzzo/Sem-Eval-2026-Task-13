import os
import random
import logging
import pandas as pd
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from torch.utils.data import Dataset

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
    Optimized PyTorch Dataset with Human-Style Structural Noise.
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
        self.mask_token_id = tokenizer.mask_token_id

    def __len__(self) -> int:
        return len(self.data)

    def _structural_noise(self, code: str) -> str:
        """
        ARMA SEGRETA PER OOD:
        Introduce imperfezioni umane (spazi, newline) per rompere 
        i pattern troppo perfetti delle AI.
        """
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            # 1. Human Hesitation: Aggiunge righe vuote (8% prob)
            if random.random() < 0.08:
                new_lines.append("")
            
            # 2. Trailing Whitespace: Sporca la fine riga (5% prob)
            if random.random() < 0.05:
                line = line + " " * random.randint(1, 3)
            
            # 3. Indentation Noise: Spazio extra a inizio riga (1% prob)
            if random.random() < 0.01:
                line = " " + line

            new_lines.append(line)
        
        return "\n".join(new_lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        code = str(self.data.at[idx, "code"])
        label = int(self.data.at[idx, "label"])
        lang_str = str(self.data.at[idx, "language"]).lower()
        
        lang_id = self.language_map.get(lang_str, -1)
        
        # ---------------------------------------------------------
        # 1. Structural Augmentation (Solo in training!)
        # ---------------------------------------------------------
        if self.augment:
            code = self._structural_noise(code)

        # ---------------------------------------------------------
        # 2. Random Cropping Strategy
        # ---------------------------------------------------------
        # Prende una finestra casuale invece che solo l'inizio
        if self.augment and len(code) > self.max_length * 4:
            # Lascia un margine alla fine
            max_start = len(code) - int(self.max_length * 3.5)
            if max_start > 0:
                start_idx = random.randint(0, max_start)
                code = code[start_idx : start_idx + int(self.max_length * 4)]

        # ---------------------------------------------------------
        # 3. Tokenization
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
        # 4. Token Masking
        # ---------------------------------------------------------
        if self.augment:
            probability_matrix = torch.full(input_ids.shape, 0.15)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                input_ids.tolist(), already_has_special_tokens=True
            )
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            probability_matrix.masked_fill_(attention_mask == 0, value=0.0)
            
            masked_indices = torch.bernoulli(probability_matrix).bool()
            input_ids[masked_indices] = self.mask_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long)
        }

# ... (Le funzioni load_and_preprocess e balance_languages restano uguali a prima)
def load_and_preprocess(file_path: str, max_code_chars: int = 20000) -> pd.DataFrame:
    columns = ['code', 'label', 'language']
    try:
        df = pd.read_parquet(file_path, columns=columns)
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise e
    df['language'] = df['language'].str.lower()
    df = df.dropna(subset=['code']).reset_index(drop=True)
    df = df[df['code'].str.len() > 20].copy()
    df['code'] = df['code'].str.slice(0, max_code_chars)
    return df

def balance_languages(df: pd.DataFrame, target_samples: int = 3000) -> pd.DataFrame:
    logger.info(f"Balancing languages... Target cap: {target_samples}")
    df_list = []
    for lang, count in df['language'].value_counts().items():
        group = df[df['language'] == lang]
        if count > target_samples:
            df_list.append(group.sample(n=target_samples, random_state=42))
        else:
            df_list.append(group)
    return pd.concat(df_list).sample(frac=1, random_state=42).reset_index(drop=True)

def load_data(config: dict, tokenizer) -> Tuple[CodeDataset, CodeDataset, pd.DataFrame, pd.DataFrame]:
    logger.info(">>> Loading Data for Task A <<<")
    train_df = load_and_preprocess(config["data"]["train_path"])
    val_df   = load_and_preprocess(config["data"]["val_path"])
    
    samples_cap = config["data"].get("samples_per_lang", 4000)
    if config["data"].get("balance_languages", True):
        train_df = balance_languages(train_df, target_samples=samples_cap)
    
    target_langs = config["model"].get("languages", ["python", "java", "cpp"])
    language_map = {lang.lower(): i for i, lang in enumerate(target_langs)}
    
    max_len = config["data"].get("max_length", 512)
    
    train_dataset = CodeDataset(train_df, tokenizer, language_map, max_length=max_len, augment=True)
    val_dataset = CodeDataset(val_df, tokenizer, language_map, max_length=max_len, augment=False)

    logger.info(f"Datasets Ready. Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    return train_dataset, val_dataset, train_df, val_df

# -----------------------------------------------------------------------------
# Main Data Loading Interface
# -----------------------------------------------------------------------------
def load_data(config: dict, tokenizer) -> Tuple[CodeDataset, CodeDataset, pd.DataFrame, pd.DataFrame]:
    logger.info(">>> Loading Data for Task A <<<")
    
    train_path = config["data"]["train_path"]
    val_path = config["data"]["val_path"]
    
    train_df = load_and_preprocess(train_path)
    val_df   = load_and_preprocess(val_path)
    
    # Balancing
    samples_cap = config["data"].get("samples_per_lang", 4000)
    if config["data"].get("balance_languages", True):
        train_df = balance_languages(train_df, target_samples=samples_cap)
    
    # DANN Mapping
    target_langs = config["model"].get("languages", ["python", "java", "cpp"])
    language_map = {lang.lower(): i for i, lang in enumerate(target_langs)}
    logger.info(f"DANN Language Map Size: {len(language_map)}")
    
    max_len = config["data"].get("max_length", 512)
    
    train_dataset = CodeDataset(train_df, tokenizer, language_map, max_length=max_len, augment=True)
    val_dataset = CodeDataset(val_df, tokenizer, language_map, max_length=max_len, augment=False)

    return train_dataset, val_dataset, train_df, val_df