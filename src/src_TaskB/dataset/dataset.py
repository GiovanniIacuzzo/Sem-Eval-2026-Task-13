import torch
import pandas as pd
import numpy as np
import os
import random
import math
import collections
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

LANG_MAP = {
    "python": 0, "java": 1, "cpp": 2, "c": 3, "c#": 4, "cs": 4, 
    "javascript": 5, "php": 6, "ruby": 7, "rust": 8, "go": 9,
    "typescript": 10, "kotlin": 11, "swift": 12, "scala": 13, "shell": 14,
    "bash": 14, "sh": 14, "c++": 2, "js": 5
}

# -----------------------------------------------------------------------------
# 1. Feature Engineering (Stylometry)
# -----------------------------------------------------------------------------
def calculate_entropy(text):
    if not text: return 0.0
    counter = collections.Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def extract_stylometric_features(code):
    """
    Estrae 5 feature scalari che differenziano Umani da AI:
    1. Lunghezza codice (log)
    2. Lunghezza media delle linee
    3. Ratio caratteri speciali (codice denso vs verboso)
    4. Entropia dei caratteri (casualità dei nomi variabili)
    5. Ratio spazi bianchi (formattazione)
    """
    code_len = len(code)
    if code_len == 0:
        return [0.0] * 5
    
    lines = code.split('\n')
    num_lines = len(lines)
    avg_line_len = code_len / max(1, num_lines)
    
    special_chars = sum(1 for c in code if not c.isalnum() and not c.isspace())
    special_ratio = special_chars / code_len
    
    entropy = calculate_entropy(code)
    
    white_space_ratio = code.count(' ') / code_len

    return [
        math.log(code_len + 1), 
        avg_line_len, 
        special_ratio, 
        entropy, 
        white_space_ratio
    ]

# -----------------------------------------------------------------------------
# 2. Augmentation
# -----------------------------------------------------------------------------
def augment_code_safe(code: str) -> str:
    if random.random() > 0.7:
        return code

    lines = code.split('\n')
    
    if random.random() > 0.5 and len(lines) > 2:
        idx = random.randint(1, len(lines)-1)
        lines.insert(idx, "")
    
    if random.random() > 0.5 and len(lines) > 0:
        idx = random.randint(0, len(lines)-1)
        lines[idx] += " "

    return "\n".join(lines)

# -----------------------------------------------------------------------------
# 3. Dataset Class
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, mode="binary", 
                 is_train=False, feature_scaler=None):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.is_train = is_train
        
        self.codes = dataframe['code'].astype(str).tolist()
        
        if self.mode == "binary":
            self.labels = dataframe['is_ai'].astype(int).tolist()
        elif self.mode == "families":
            self.labels = dataframe['family_label'].astype(int).tolist()
        else:
            self.labels = dataframe['label'].astype(int).tolist()

        if 'language' not in dataframe.columns:
            dataframe['language'] = 'unknown'
        
        self.lang_ids = dataframe['language'].apply(
            lambda x: LANG_MAP.get(str(x).lower().strip(), -1)
        ).tolist()

        print(f"[{'Train' if is_train else 'Val'}] Extracting Stylometric Features...")
        features_list = [extract_stylometric_features(c) for c in self.codes]
        self.features = np.array(features_list, dtype=np.float32)

        if is_train:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            if feature_scaler is None:
                raise ValueError("Validation set needs a fitted scaler from training set!")
            self.scaler = feature_scaler
            self.features = self.scaler.transform(self.features)

    def __len__(self):
        return len(self.codes)

    def get_scaler(self):
        return self.scaler

    def __getitem__(self, idx):
        code = self.codes[idx]
        label = self.labels[idx]
        lang_id = self.lang_ids[idx]
        extra_feat = self.features[idx]

        if self.is_train:
            code = augment_code_safe(code)

        tokens = self.tokenizer(code, truncation=False, padding=False, return_tensors=None)["input_ids"]
        
        if len(tokens) > self.max_length:
            half_len = (self.max_length - 2) // 2
            head = tokens[:half_len]
            tail = tokens[-half_len:]
            input_ids = head + tail
        else:
            input_ids = tokens

        encoding = self.tokenizer.prepare_for_model(
            input_ids,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long),
            "extra_features": torch.tensor(extra_feat, dtype=torch.float)
        }

# -----------------------------------------------------------------------------
# Loader Function
# -----------------------------------------------------------------------------
def load_data(config, tokenizer, mode="binary"):
    common_cfg = config["data"]
    data_dir = common_cfg.get("data_dir", "data/Task_B_Processed")
    
    train_file = f"train_{mode}.parquet"
    val_file = f"val_{mode}.parquet"
    
    train_path = os.path.join(data_dir, train_file)
    val_path = os.path.join(data_dir, val_file)
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing data files in {data_dir}")

    print(f"Loading Parquet files from {data_dir}...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    class_weights_tensor = None
    if mode == "families" and config["model"].get("class_weights", False):
        print("Computing Class Weights (Balanced)...")
        y_train = train_df['family_label'].values
        classes = np.unique(y_train)
        cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights_tensor = torch.tensor(cw, dtype=torch.float)

    train_ds = CodeDataset(
        train_df, 
        tokenizer, 
        max_length=common_cfg["max_length"], 
        mode=mode, 
        is_train=True,
        feature_scaler=None 
    )
    
    train_scaler = train_ds.get_scaler()

    val_ds = CodeDataset(
        val_df, 
        tokenizer, 
        max_length=common_cfg["max_length"], 
        mode=mode, 
        is_train=False,
        feature_scaler=train_scaler 
    )

    return train_ds, val_ds, class_weights_tensor