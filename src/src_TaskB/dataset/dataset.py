import torch
import pandas as pd
import numpy as np
import os
import re
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, mode="binary", is_train=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.is_train = is_train
        
        self.samples = []
        
        codes = dataframe['code'].astype(str).tolist()
        
        if self.mode == "binary":
            labels = dataframe['is_ai'].astype(int).tolist()
        elif self.mode == "families":
            labels = dataframe['family_label'].astype(int).tolist()
        
        stride = 256
        print(f"[{'Train' if is_train else 'Val'}] Processing {len(codes)} files with Stylistic Extraction...")
        
        for i, code in enumerate(codes):
            label = labels[i]
            stylistic_feats = self._extract_stylistic_features(code)
            
            tokens = tokenizer.tokenize(code)
            capacity = max_length - 2
            
            if len(tokens) <= capacity:
                self.samples.append({
                    "code_str": code,
                    "label": label,
                    "extra_features": stylistic_feats
                })
            else:
                if is_train:
                    for start_idx in range(0, len(tokens), stride):
                        end_idx = min(start_idx + capacity, len(tokens))
                        chunk_tokens = tokens[start_idx:end_idx]
                        chunk_str = tokenizer.convert_tokens_to_string(chunk_tokens)
                        
                        self.samples.append({
                            "code_str": chunk_str,
                            "label": label,
                            "extra_features": stylistic_feats
                        })
                        if end_idx >= len(tokens): break
                else:
                    half = capacity // 2
                    head = tokens[:half]
                    tail = tokens[-half:]
                    chunk_str = tokenizer.convert_tokens_to_string(head + tail)
                    self.samples.append({
                        "code_str": chunk_str,
                        "label": label,
                        "extra_features": stylistic_feats
                    })

        print(f"[{'Train' if is_train else 'Val'}] Total Samples: {len(self.samples)}")

    def _extract_stylistic_features(self, code):
        """
        Estrae un vettore di 8 caratteristiche che definiscono la 'firma' dell'LLM.
        Questi valori vengono normalizzati per aiutare la convergenza.
        """
        features = []
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        code_len = len(code) + 1
        
        features.append(code.count(' ') / code_len)
        
        comment_chars = code.count('#') + code.count('//') + (code.count('/*') * 2)
        features.append(comment_chars / code_len)
        
        special_chars = len(re.findall(r'[{}()\[\];.,]', code))
        features.append(special_chars / code_len)
        
        avg_line_len = np.mean([len(l) for l in non_empty_lines]) if non_empty_lines else 0
        features.append(min(avg_line_len / 100.0, 1.0))
        
        empty_ratio = (len(lines) - len(non_empty_lines)) / (len(lines) + 1)
        features.append(empty_ratio)
        
        snake_count = code.count('_')
        camel_count = len(re.findall(r'[a-z][A-Z]', code))
        features.append(snake_count / (snake_count + camel_count + 1))
        
        logic_tokens = len(re.findall(r'\b(if|for|while|return|switch|case|break)\b', code))
        features.append(logic_tokens / (len(code.split()) + 1))
        
        max_indent = max([len(l) - len(l.lstrip()) for l in non_empty_lines]) if non_empty_lines else 0
        features.append(min(max_indent / 20.0, 1.0))
        
        return torch.tensor(features, dtype=torch.float)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample["code_str"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
            "extra_features": sample["extra_features"] 
        }

def load_data(config, tokenizer, mode="binary"):
    common_cfg = config["data"]
    data_dir = common_cfg.get("data_dir", "data/Task_B_Processed")
    
    train_path = os.path.join(data_dir, f"train_{mode}.parquet")
    val_path = os.path.join(data_dir, f"val_{mode}.parquet")
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    class_weights_tensor = None
    if mode == "families":
        y_train = train_df['family_label'].values
        classes = np.unique(y_train)
        cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights_tensor = torch.tensor(cw, dtype=torch.float)

    train_ds = CodeDataset(train_df, tokenizer, max_length=common_cfg["max_length"], mode=mode, is_train=True)
    val_ds = CodeDataset(val_df, tokenizer, max_length=common_cfg["max_length"], mode=mode, is_train=False)

    return train_ds, val_ds, class_weights_tensor