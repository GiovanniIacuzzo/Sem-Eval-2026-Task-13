import torch
import pandas as pd
import numpy as np
import os
import re
import logging
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, mode="binary", is_train=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.is_train = is_train
        
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.extra_features = []
        
        codes = dataframe['code'].astype(str).tolist()
        if self.mode == "binary":
            raw_labels = dataframe['is_ai'].astype(int).tolist()
        else:
            raw_labels = dataframe['family_label'].astype(int).tolist()
        
        stride = 256
        desc = f"[{'Train' if is_train else 'Val'}] Tokenizing & Extracting"
        
        for i, code in enumerate(tqdm(codes, desc=desc, leave=False)):
            label = raw_labels[i]
            stylistic_feats = self._extract_stylistic_features(code)
            
            tokens = tokenizer.tokenize(code)
            capacity = max_length - 2
            
            processed_chunks = []
            if len(tokens) <= capacity:
                processed_chunks.append(code)
            else:
                if is_train:
                    for count, start_idx in enumerate(range(0, len(tokens), stride)):
                        if count >= 3: break 
                        end_idx = min(start_idx + capacity, len(tokens))
                        chunk_str = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx])
                        processed_chunks.append(chunk_str)
                else:
                    half = capacity // 2
                    chunk_str = tokenizer.convert_tokens_to_string(tokens[:half] + tokens[-half:])
                    processed_chunks.append(chunk_str)

            for chunk in processed_chunks:
                encoding = self.tokenizer(
                    chunk,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                self.input_ids.append(encoding["input_ids"].squeeze(0))
                self.attention_masks.append(encoding["attention_mask"].squeeze(0))
                self.labels.append(torch.tensor(label, dtype=torch.long))
                self.extra_features.append(stylistic_feats)

        print(f"[{'Train' if is_train else 'Val'}] Final Samples: {len(self.labels)}")

    def _extract_stylistic_features(self, code):
        features = []
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        code_len = len(code) + 1
        
        features.append(code.count(' ') / code_len)
        features.append((code.count('#') + code.count('//')) / code_len)
        features.append(len(re.findall(r'[{}()\[\];.,]', code)) / code_len)
        
        avg_line_len = np.mean([len(l) for l in non_empty_lines]) if non_empty_lines else 0
        features.append(min(avg_line_len / 100.0, 1.0))
        features.append((len(lines) - len(non_empty_lines)) / (len(lines) + 1))
        
        snake_count = code.count('_')
        camel_count = len(re.findall(r'[a-z][A-Z]', code))
        features.append(snake_count / (snake_count + camel_count + 1))
        
        logic_tokens = len(re.findall(r'\b(if|for|while|return|switch|case|break)\b', code))
        features.append(logic_tokens / (len(code.split()) + 1))
        
        max_indent = max([len(l) - len(l.lstrip()) for l in non_empty_lines]) if non_empty_lines else 0
        features.append(min(max_indent / 20.0, 1.0))
        
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

def load_data(config, tokenizer, mode="binary"):
    data_dir = config["data"].get("data_dir", "data/Task_B_Processed")
    
    if mode == "binary":
        train_df = pd.read_parquet(os.path.join(data_dir, "train_binary.parquet"))
        val_df = pd.read_parquet(os.path.join(data_dir, "val_binary.parquet"))
        
        # Logica di downsampling per Binary (OK quella che avevi)
        df_ai = train_df[train_df['is_ai'] == 1]
        df_human = train_df[train_df['is_ai'] == 0].sample(n=min(len(df_ai), 40000), random_state=42)
        train_df = pd.concat([df_ai, df_human]).sample(frac=1, random_state=42)
        
    elif mode == "families":
        # CARICHIAMO SOLO IL DATASET FILTRATO AI
        train_df = pd.read_parquet(os.path.join(data_dir, "train_families.parquet"))
        val_df = pd.read_parquet(os.path.join(data_dir, "val_families.parquet"))
        
        logger.info(f"Family Training Samples (AI Only): {len(train_df)}")
        logger.info(f"Family Validation Samples (AI Only): {len(val_df)}")
        
        # Verifica di sicurezza
        if 'family_label' not in train_df.columns:
            raise ValueError("train_families.parquet non ha la colonna 'family_label'")

    # Calcolo pesi classi
    class_weights_tensor = None
    if mode == "families" and config["model"].get("class_weights", False):
        y_train = train_df['family_label'].values
        # Usa labels univoche presenti nel mapping (0..10)
        classes = np.arange(11) 
        cw = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        # Mappa i pesi su un tensore di dimensione fissa 11 per evitare errori se una classe manca nel train
        weights_full = np.ones(11)
        for cls, w in zip(np.unique(y_train), cw):
            weights_full[cls] = w
        class_weights_tensor = torch.tensor(weights_full, dtype=torch.float)

    train_ds = CodeDataset(train_df, tokenizer, max_length=config["data"]["max_length"], mode=mode, is_train=True)
    val_ds = CodeDataset(val_df, tokenizer, max_length=config["data"]["max_length"], mode=mode, is_train=False)

    return train_ds, val_ds, class_weights_tensor