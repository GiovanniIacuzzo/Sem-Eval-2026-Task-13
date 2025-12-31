import torch
import pandas as pd
import numpy as np
import logging
import random
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, language_map, max_length=512, is_test=False, augment=False):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.language_map = language_map
        self.max_length = max_length
        self.is_test = is_test
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def _structural_noise(self, code):
        if not self.augment: return code
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            if random.random() < 0.1: new_lines.append("") 
            if random.random() < 0.1: line = line + " " * random.randint(1, 4)
            if random.random() < 0.01: line = " " + line 
            new_lines.append(line)
        return "\n".join(new_lines)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        raw_code = str(row['code']) if row['code'] is not None else ""
        code = self._structural_noise(raw_code)
        
        lang_str = str(row['language']).lower() if 'language' in row else 'unknown'
        lang_id = self.language_map.get(lang_str, 0) 

        encoding = self.tokenizer(code, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long)
        }
        if not self.is_test:
            label = int(row['label'])
            item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

def get_dann_class_weights(df, language_map, device):
    """Calcola i pesi inversi per la loss DANN."""
    valid_langs = [l for l in df['language'] if l in language_map]
    if not valid_langs: return None
    
    y_langs = [language_map[l] for l in valid_langs]
    classes = np.unique(y_langs)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_langs)
    
    weight_tensor = torch.ones(len(language_map)).to(device)
    for cls_idx, w in zip(classes, weights):
        weight_tensor[cls_idx] = w
    return weight_tensor

def balance_dataframe(df, config):
    """Funzione helper per bilanciare un dataframe specifico (es. train fold)."""
    lang_counts = df['language'].value_counts()
    if len(lang_counts) > 1:
        target_per_lang = int(lang_counts.iloc[1] * 1.5) 
    else:
        target_per_lang = 10000 
        
    global_cap = config["data"].get("max_samples_per_language", 15000)
    target_samples = min(target_per_lang, global_cap)
    
    balanced_dfs = []
    for (lang, label), group in df.groupby(['language', 'label']):
        if len(group) > (target_samples // 2): 
            balanced_dfs.append(group.sample(n=(target_samples // 2), random_state=42))
        else:
            balanced_dfs.append(group)
            if len(group) < 1000: # Oversampling per classi minuscole
                balanced_dfs.append(group.sample(n=1000, replace=True, random_state=42))

    return pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

def load_data_raw(config):
    """Carica i dati GREZZI senza bilanciare."""
    logger.info("Loading Raw Datasets...")
    df_train = pd.read_parquet(config["data"]["train_path"])
    df_val = pd.read_parquet(config["data"]["val_path"])
    
    for df in [df_train, df_val]:
        if 'original_code' in df.columns: df.rename(columns={'original_code': 'code'}, inplace=True)
        if 'original_lang' in df.columns: df.rename(columns={'original_lang': 'language'}, inplace=True)
        df['language'] = df['language'].str.lower()
        
    return df_train, df_val