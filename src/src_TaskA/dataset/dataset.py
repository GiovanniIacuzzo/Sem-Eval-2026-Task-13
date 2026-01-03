import torch
import pandas as pd
import numpy as np
import logging
import random
import re
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
        
        self.keywords = {
            'python': {'def', 'import', 'from', 'class', 'self', 'elif', 'None', 'print'},
            'java': {'public', 'static', 'void', 'private', 'protected', 'new', 'final', 'class'},
            'cpp': {'include', 'std', 'cout', 'vector', 'using', 'namespace', 'template', 'void'}
        }

    def _mask_keywords(self, code, lang):
        lang_keywords = self.keywords.get(lang.lower(), set())
        for kw in lang_keywords:
            if random.random() < 0.4:
                code = re.sub(r'\b' + re.escape(kw) + r'\b', self.tokenizer.mask_token, code)
        return code

    def _apply_augmentation(self, code, lang):
        code = re.sub(r'//.*|#.*|/\*.*?\*/', '', code, flags=re.DOTALL)
        
        if not self.augment: return code

        if random.random() < 0.6:
            words = re.findall(r'\b[a-z_][a-z0-9_]{3,}\b', code)
            for w in set(words):
                if random.random() < 0.7:
                    code = re.sub(r'\b'+w+r'\b', f"var_{random.randint(0,99)}", code)

        if random.random() < 0.5:
            code = self._mask_keywords(code, lang)
            
        return code

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        lang_str = str(row['language']).lower()
        code = self._apply_augmentation(str(row['code']), lang_str)
        
        encoding = self.tokenizer(
            code, truncation=True, padding="max_length", 
            max_length=self.max_length, return_tensors="pt"
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "lang_ids": torch.tensor(self.language_map.get(lang_str, 0), dtype=torch.long)
        }
        if not self.is_test:
            item["labels"] = torch.tensor(int(row['label']), dtype=torch.long)
        return item

def load_data_raw(config):
    """Carica i file parquet e uniforma le colonne."""
    logger.info("Loading Raw Datasets...")
    df_train = pd.read_parquet(config["data"]["train_path"])
    df_val = pd.read_parquet(config["data"]["val_path"])
    
    for df in [df_train, df_val]:
        if 'original_code' in df.columns: df.rename(columns={'original_code': 'code'}, inplace=True)
        if 'original_lang' in df.columns: df.rename(columns={'original_lang': 'language'}, inplace=True)
        df['language'] = df['language'].str.lower().fillna('unknown')
        
    return df_train, df_val

def balance_dataframe(df, config):
    """Bilancia il dataset per lingua e classe (AI/Human)."""
    max_samples = config["data"].get("max_samples_per_language", 15000)
    balanced_dfs = []
    
    for lang in df['language'].unique():
        lang_df = df[df['language'] == lang]
        for label in [0, 1]:
            subset = lang_df[lang_df['label'] == label]
            if len(subset) == 0: continue
            
            n_samples = min(len(subset), max_samples)
            replace = True if len(subset) < 2000 else False
            if replace: n_samples = max(n_samples, 2000)
            
            balanced_dfs.append(subset.sample(n=int(n_samples), replace=replace, random_state=42))

    return pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

def get_dann_class_weights(df, language_map, device):
    """Calcola i pesi per il classificatore di lingua (DANN)."""
    valid_langs = [l for l in df['language'] if l in language_map]
    if not valid_langs: return None
    
    y_langs = [language_map[l] for l in valid_langs]
    classes = np.unique(y_langs)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_langs)
    
    weight_tensor = torch.ones(len(language_map)).to(device)
    for cls_idx, w in zip(classes, weights):
        weight_tensor[cls_idx] = torch.tensor(w, dtype=torch.float)
    return weight_tensor