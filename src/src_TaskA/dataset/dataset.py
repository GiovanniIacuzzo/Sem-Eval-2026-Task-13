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

    def __len__(self):
        return len(self.data)

    def _remove_comments(self, code):
        # Rimuove commenti stile C (//) e Python/Ruby (#)
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'#.*', '', code)
        # Rimuove commenti multi-riga /* ... */
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code

    def _rename_variables(self, code):
        keywords = {'if', 'else', 'for', 'while', 'return', 'def', 'class', 'import', 'from', 'public', 'static', 'void', 'int', 'float'}
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        
        unique_vars = sorted(list(set([w for w in words if w not in keywords and len(w) > 2])))
        
        var_map = {v: f"VAR_{i}" for i, v in enumerate(unique_vars)}
        
        for original, replacement in var_map.items():
            code = re.sub(r'\b' + re.escape(original) + r'\b', replacement, code)
        return code

    def _apply_augmentation(self, code):
        if not self.augment:
            return code
        
        # 1. Rimuovi commenti
        if random.random() < 0.7:
            code = self._remove_comments(code)
        
        # 2. Rinomina variabili
        if random.random() < 0.4:
            code = self._rename_variables(code)
            
        # 3. Rumore strutturale
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            if not line.strip(): continue
            if random.random() < 0.05: new_lines.append("")
            if random.random() < 0.1: 
                line = line + (" " * random.randint(1, 3))
            new_lines.append(line)
            
        return "\n".join(new_lines)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        code = str(row['code']) if row['code'] is not None else ""
        
        if self.augment:
            code = self._apply_augmentation(code)
        
        lang_str = str(row['language']).lower() if 'language' in row else 'unknown'
        lang_id = self.language_map.get(lang_str, 0) 

        encoding = self.tokenizer(
            code, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long)
        }
        
        if not self.is_test:
            item["labels"] = torch.tensor(int(row['label']), dtype=torch.long)
            
        return item

def get_dann_class_weights(df, language_map, device):
    valid_langs = [l for l in df['language'] if l in language_map]
    if not valid_langs: return None
    
    y_langs = [language_map[l] for l in valid_langs]
    classes = np.unique(y_langs)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_langs)
    
    weight_tensor = torch.ones(len(language_map)).to(device)
    for cls_idx, w in zip(classes, weights):
        weight_tensor[cls_idx] = torch.tensor(w, dtype=torch.float)
    return weight_tensor

def balance_dataframe(df, config):
    max_samples = config["data"].get("max_samples_per_language", 15000)
    
    balanced_dfs = []
    for lang in df['language'].unique():
        lang_df = df[df['language'] == lang]
        for label in [0, 1]:
            subset = lang_df[lang_df['label'] == label]
            if len(subset) == 0: continue
            
            n_samples = min(len(subset), max_samples)
            
            replace = True if len(subset) < 2000 else False
            if replace: n_samples = 2000
            
            balanced_dfs.append(subset.sample(n=n_samples, replace=replace, random_state=42))

    return pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

def load_data_raw(config):
    logger.info("Loading Raw Datasets...")
    df_train = pd.read_parquet(config["data"]["train_path"])
    df_val = pd.read_parquet(config["data"]["val_path"])
    
    for df in [df_train, df_val]:
        if 'original_code' in df.columns: df.rename(columns={'original_code': 'code'}, inplace=True)
        if 'original_lang' in df.columns: df.rename(columns={'original_lang': 'language'}, inplace=True)
        df['language'] = df['language'].str.lower().fillna('unknown')
        
    return df_train, df_val