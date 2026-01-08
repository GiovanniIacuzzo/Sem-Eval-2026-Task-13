import torch
import pandas as pd
import logging
import math
import os
import re
import collections
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# =============================================================================
# 1. DATASET CLASS
# =============================================================================
class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, lang2id, max_length=512, feature_stats=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang2id = lang2id
        
        dataframe = dataframe.reset_index(drop=True)
        
        self.codes = dataframe['code'].astype(str).tolist()
        self.labels = dataframe['label'].astype(int).tolist()
        self.languages = dataframe['language'].astype(str).tolist()
        
        self.lang_ids = [self.lang2id[l] if l in self.lang2id else -1 for l in self.languages]

        if 'perplexity' in dataframe.columns:
            self.perplexities = dataframe['perplexity'].astype(float).tolist()
        else:
            self.perplexities = [0.0] * len(self.codes)
            logger.warning("[Dataset] 'perplexity' column missing. Using 0.0 (Feature disabled).")

        logger.info(f"[Dataset] Extracting features for {len(self.codes)} samples...")
        self.raw_features = []
        
        for idx, code in enumerate(tqdm(self.codes, desc="Extracting")):
            self.raw_features.append(self._extract_features(code, idx))
            
        self.features_tensor = torch.stack(self.raw_features)
        
        if feature_stats is None:
            self.mean = self.features_tensor.mean(dim=0)
            self.std = self.features_tensor.std(dim=0) + 1e-6
            self.stats = {'mean': self.mean, 'std': self.std}
            logger.info(f"[Dataset] Computed Stats -> Mean: {self.mean[:2]}... | Std: {self.std[:2]}...")
        else:
            self.mean = feature_stats['mean']
            self.std = feature_stats['std']
            self.stats = feature_stats
            logger.info(f"[Dataset] Using External Stats -> Mean: {self.mean[:2]}...")

        # Applicazione Z-Score
        self.features_tensor = (self.features_tensor - self.mean) / self.std

    def _extract_features(self, code, idx):
        """
        Feature Stilometriche Agnostiche (Prof. Style) + Perplexity
        """
        s_code = str(code)
        if not s_code: s_code = " "
        
        features = []
        
        identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', s_code)
        keywords = {"if", "else", "for", "while", "return", "int", "float", "def", "class", "void", "import", "from", "public", "private", "var", "const", "let", "function"}
        identifiers = [x for x in identifiers if len(x) > 1 and x not in keywords]
        
        if len(identifiers) == 0:
            avg_id_len, id_entropy, snake_case_ratio, camel_case_ratio = 0.0, 0.0, 0.0, 0.0
        else:
            # 1. Lunghezza media
            avg_id_len = sum(len(x) for x in identifiers) / len(identifiers)
            
            # 2. Entropia Caratteri
            all_ids_str = "".join(identifiers)
            counter = collections.Counter(all_ids_str)
            total_chars = len(all_ids_str)
            id_entropy = 0.0
            if total_chars > 0:
                for count in counter.values():
                    p = count / total_chars
                    id_entropy -= p * math.log2(p)
            
            # 3. Naming Mix
            snake_case = sum(1 for x in identifiers if '_' in x)
            camel_case = sum(1 for x in identifiers if '_' not in x and any(c.isupper() for c in x))
            snake_case_ratio = snake_case / len(identifiers)
            camel_case_ratio = camel_case / len(identifiers)

        features.append(math.log1p(avg_id_len))
        features.append(id_entropy)
        features.append(snake_case_ratio)
        features.append(camel_case_ratio)

        # 4. Unique Token Ratio
        tokens = s_code.split()
        if len(tokens) == 0: unique_ratio = 0.0
        else: unique_ratio = len(set(tokens)) / len(tokens)
        features.append(unique_ratio)

        # 5. Struttura Visiva
        lines = s_code.split('\n')
        num_lines = len(lines) + 1.0
        empty_lines = sum(1 for line in lines if not line.strip())
        features.append(empty_lines / num_lines)
        
        # 6. Dirty Markers
        dirty_markers = ["todo", "fix", "hack", "???", "!!!", "temp", "tmp"]
        s_code_lower = s_code.lower()
        dirty_count = sum(s_code_lower.count(m) for m in dirty_markers)
        features.append(math.log1p(dirty_count))
        
        # 7. PERPLEXITY
        ppl_val = self.perplexities[idx]
        features.append(math.log1p(ppl_val))

        return torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.codes[idx],
            return_tensors=None, 
            max_length=self.max_length,
            truncation=True,
            padding=False
        )
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": self.labels[idx],
            "language_labels": self.lang_ids[idx],
            "extra_features": self.features_tensor[idx]
        }

# =============================================================================
# 2. LOADING FUNCTION
# =============================================================================
def load_data_lolo(config, tokenizer, holdout_language=None):
    data_dir = config["data_dir"]
    logger.info("Loading Dataframes...")
    
    try:
        train_path = os.path.join(data_dir, "train_ppl.parquet")
        val_path = os.path.join(data_dir, "validation_ppl.parquet")
        
        if os.path.exists(train_path) and os.path.exists(val_path):
            logger.info("Found pre-computed Perplexity files. Loading...")
            df1 = pd.read_parquet(train_path)
            df2 = pd.read_parquet(val_path)
        else:
            raise FileNotFoundError("PPL files not found.")
    except (FileNotFoundError, Exception):
        logger.warning("Perplexity files not found! Loading original files (Feature 8 will be 0).")
        df1 = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
        df2 = pd.read_parquet(os.path.join(data_dir, "validation.parquet"))
    
    # Concatenazione Iniziale
    full_df = pd.concat([df1, df2], ignore_index=True)

    # Normalizzazione Linguaggi
    full_df['language'] = full_df['language'].astype(str).str.lower().str.strip()
    available_langs = sorted(full_df['language'].unique().tolist())
    
    # Mappa ID
    lang2id = {lang: idx for idx, lang in enumerate(available_langs)}
    logger.info(f"Language Map: {lang2id}")

    # Logica LOLO
    if holdout_language:
        holdout_language = holdout_language.lower().strip()
        if holdout_language not in available_langs:
            raise ValueError(f"Holdout Language '{holdout_language}' not found in {available_langs}")
            
        logger.info(f"STRATEGIA LOLO: Holdout on {holdout_language}")
        val_df = full_df[full_df['language'] == holdout_language].copy()

        train_df = full_df[full_df['language'] != holdout_language].copy()
    else:
        logger.warning("Split Random (Standard)")
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(full_df, test_size=0.1, random_state=42, stratify=full_df['label'])

    # Data Balancing 
    if config.get("balance_languages", True):
        target = config.get("target_size_per_language", 25000)
        train_chunks = []
        for lang in train_df['language'].unique():
            lang_df = train_df[train_df['language'] == lang]
            half = target // 2
            
            for lbl in [0, 1]:
                sub = lang_df[lang_df['label'] == lbl]
                if len(sub) > 0:
                    train_chunks.append(sub.sample(n=half, replace=(len(sub)<half), random_state=42))
        
        train_df = pd.concat(train_chunks).sample(frac=1, random_state=42).reset_index(drop=True)

    # 1. Train (Calcola Stats)
    train_ds = CodeDataset(
        train_df, 
        tokenizer, 
        lang2id, 
        max_length=config.get("max_length", 512), 
        feature_stats=None 
    )
    
    # 2. Val (Usa Stats Train)
    val_ds = CodeDataset(
        val_df, 
        tokenizer, 
        lang2id, 
        max_length=config.get("max_length", 512), 
        feature_stats=train_ds.stats 
    )
    
    return train_ds, val_ds, lang2id