from torch.utils.data import Dataset
import numpy as np
import torch
import math
import re
from collections import Counter

class InferenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length        
        self.codes = dataframe['code'].astype(str).tolist()
        self.labels = dataframe['label'].astype(int).tolist() if 'label' in dataframe.columns else [-1] * len(self.codes)
        
        # Gestione Feature
        if 'extra_features' in dataframe.columns:
            self.extra_features = dataframe['extra_features'].tolist()
            self.has_precomputed = True
        else:
            self.extra_features = None
            self.has_precomputed = False
            
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self):
        return len(self.codes)

    def _extract_robust_features(self, code):
        """Stessa logica usata in training per consistenza"""
        features = []
        code_len = len(code) + 1
        
        counts = Counter(code)
        entropy = -sum((cnt / code_len) * math.log2(cnt / code_len) for cnt in counts.values())
        features.append(entropy / 8.0)
        
        specials = len(re.findall(r'[{}()\[\];.,+\-*/%&|^!=<>?]', code))
        features.append(specials / code_len)
        
        features.append(code.count(' ') / code_len)
        
        words = re.findall(r'\w+', code)
        num_words = len(words)
        if num_words > 0:
            avg_word_len = sum(len(w) for w in words) / num_words
            unique_ratio = len(set(words)) / num_words
        else:
            avg_word_len = 0
            unique_ratio = 0
            
        features.append(min(avg_word_len / 20.0, 1.0))
        features.append(len(re.findall(r'\b(if|for|while|return|def|class|import|void|int)\b', code)) / (num_words + 1))
        features.append(unique_ratio)
        features.append(min(len(re.findall(r'"[^"]{50,}"|\'[^\']{50,}\'', code)) / 5.0, 1.0))
        
        current_depth = 0
        max_depth = 0
        for char in code:
            if char == '{': current_depth += 1
            elif char == '}': current_depth = max(0, current_depth - 1)
            max_depth = max(max_depth, current_depth)
        features.append(min(max_depth / 10.0, 1.0))

        return torch.tensor(features, dtype=torch.float)

    def __getitem__(self, idx):
        code = self.codes[idx]
        
        # Feature Extraction
        if self.has_precomputed:
            feats = self.extra_features[idx]
            if isinstance(feats, np.ndarray): feats = feats.tolist()
            extra_feats = torch.tensor(feats, dtype=torch.float)
        else:
            extra_feats = self._extract_robust_features(code)
            
        # Tokenization
        inputs = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "extra_features": extra_feats,
            "original_label": self.labels[idx]
        }

def inference_collate(batch):
    return {
        "input_ids": torch.stack([item['input_ids'] for item in batch]),
        "attention_mask": torch.stack([item['attention_mask'] for item in batch]),
        "extra_features": torch.stack([item['extra_features'] for item in batch]),
        "original_labels": [item['original_label'] for item in batch]
    }