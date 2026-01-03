import os
import random
import logging
import pandas as pd
import torch
import numpy as np
import re
from typing import Dict
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------------------------------------------------------
# Logger Setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
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

    def _extract_stylometrics(self, code: str) -> torch.Tensor:
        lines = code.split('\n')
        num_lines = len(lines) if len(lines) > 0 else 1
        code_len = len(code) + 1
        
        comments = len(re.findall(r'(//|#|/\*|--)', code))
        comment_density = comments / (len(code.split()) + 1)
        
        space_ratio = code.count(' ') / code_len
        
        avg_line_len = (code_len / num_lines) / 100.0
        
        special_chars = len(re.findall(r'[!@#$%^&*()\-+={}\[\]|\\:;"\'<>,.?/]', code)) / code_len

        return torch.tensor([comment_density, space_ratio, avg_line_len, special_chars], dtype=torch.float32)

    def _structural_noise(self, code: str) -> str:
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            if random.random() < 0.05: new_lines.append("")
            if random.random() < 0.05: line = line + " " * random.randint(1, 3)
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

        full_tokens = self.tokenizer.encode(code, add_special_tokens=False)
        
        if len(full_tokens) > (self.max_length - 2):
            half = (self.max_length // 2) - 1
            head = full_tokens[:half]
            tail = full_tokens[-half:]
            input_ids = [self.tokenizer.cls_token_id] + head + tail + [self.tokenizer.sep_token_id]
        else:
            input_ids = self.tokenizer.encode(code, max_length=self.max_length, truncation=True, padding="max_length")

        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        input_ids = torch.tensor(input_ids[:self.max_length], dtype=torch.long)
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        style_feats = self._extract_stylometrics(code)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "style_feats": style_feats,
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long)
        }

    def __len__(self) -> int:
        return len(self.data)

def load_and_preprocess(file_path: str, max_code_chars: int = 15000) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_parquet(file_path, columns=['code', 'label', 'language'])
    df['language'] = df['language'].str.lower()
    df = df.dropna(subset=['code']).reset_index(drop=True)
    df = df[df['code'].str.len() > 10].copy()
    df['code'] = df['code'].str.slice(0, max_code_chars)
    return df

def get_smart_subset(df: pd.DataFrame, max_total: int = 200000) -> pd.DataFrame:
    logger.info(f"Applying Smart Subsetting (Target: {max_total} samples)...")
    
    df_hybrid = df[df['label'] == 2]
    df_adversarial = df[df['label'] == 3]
    
    df_human = df[df['label'] == 0]
    df_machine = df[df['label'] == 1]
    
    special_count = len(df_hybrid) + len(df_adversarial)
    remaining = max_total - special_count
    
    if remaining <= 0:
        return pd.concat([df_hybrid, df_adversarial]).sample(frac=1).reset_index(drop=True)
    
    h_sample = min(len(df_human), remaining // 2)
    m_sample = min(len(df_machine), remaining // 2)
    
    df_h_sub = df_human.sample(n=h_sample, random_state=42)
    df_m_sub = df_machine.sample(n=m_sample, random_state=42)
    
    final_df = pd.concat([df_hybrid, df_adversarial, df_h_sub, df_m_sub])
    return final_df.sample(frac=1, random_state=42).reset_index(drop=True)

def get_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    labels = df['label'].values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def get_dynamic_language_map(df: pd.DataFrame) -> Dict[str, int]:
    unique_langs = sorted(df['language'].unique())
    return {lang: i for i, lang in enumerate(unique_langs)}

def load_data_for_training(config: dict) -> pd.DataFrame:
    logger.info(">>> Loading Data with Smart Strategy <<<")
    df_train = load_and_preprocess(config["data"]["train_path"])
    df_val = load_and_preprocess(config["data"]["val_path"])
    
    # Uniamo per pulizia ma poi useremo uno split fisso
    full_df = pd.concat([df_train, df_val]).reset_index(drop=True)
    
    # Applichiamo il limite massimo di record per risparmiare risorse
    max_samples = config["data"].get("max_training_samples", 200000)
    full_df = get_smart_subset(full_df, max_total=max_samples)
    
    logger.info(f"Final training set size: {len(full_df)}")
    return full_df