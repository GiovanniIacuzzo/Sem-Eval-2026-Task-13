import os
import logging
import pandas as pd
import torch
import numpy as np
from typing import Dict, List, Iterator
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# -----------------------------------------------------------------------------
# 1. Custom Sampler
# -----------------------------------------------------------------------------
class BalancedBatchSampler(Sampler):
    def __init__(self, labels: List[int], batch_size: int):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.cls0_indices = np.where(self.labels == 0)[0]
        self.cls1_indices = np.where(self.labels == 1)[0]
        self.n_cls0 = len(self.cls0_indices)
        self.n_cls1 = len(self.cls1_indices)
        
        # Logica: Copre la classe minoritaria
        if self.n_cls0 > 0 and self.n_cls1 > 0:
            min_size = min(self.n_cls0, self.n_cls1)
            self.n_batches = int((min_size * 2) / self.batch_size)
        else:
            self.n_batches = int(len(self.labels) / self.batch_size)
        self.n_samples_per_class = self.batch_size // 2

    def __iter__(self) -> Iterator[int]:
        np.random.shuffle(self.cls0_indices)
        np.random.shuffle(self.cls1_indices)
        ptr0 = 0
        ptr1 = 0
        for _ in range(self.n_batches):
            batch_indices = []
            if self.n_cls0 > 0:
                end0 = ptr0 + self.n_samples_per_class
                if end0 > self.n_cls0:
                    np.random.shuffle(self.cls0_indices)
                    ptr0 = 0
                    end0 = self.n_samples_per_class
                batch_indices.extend(self.cls0_indices[ptr0:end0])
                ptr0 = end0
            if self.n_cls1 > 0:
                end1 = ptr1 + (self.batch_size - len(batch_indices))
                if end1 > self.n_cls1:
                    np.random.shuffle(self.cls1_indices)
                    ptr1 = 0
                    end1 = self.batch_size - len(batch_indices)
                batch_indices.extend(self.cls1_indices[ptr1:end1])
                ptr1 = end1
            np.random.shuffle(batch_indices)
            for idx in batch_indices:
                yield int(idx)

    def __len__(self) -> int:
        return self.n_batches * self.batch_size

# -----------------------------------------------------------------------------
# 2. Dataset Class
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, language_map: Dict[str, int], max_length: int = 512, augment: bool = False):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.language_map = language_map
        self.max_length = max_length
        self.augment = augment
        
        try:
            from src.src_TaskA.features.stylometry import StylometryExtractor
            self.stylo_extractor = StylometryExtractor()
        except ImportError:
            self.stylo_extractor = None

        self.labels_list = self.data['label'].astype(int).tolist()
        self.has_paired_data = 'aug_code' in self.data.columns

    def __len__(self) -> int:
        return len(self.data)

    def _clean_code(self, code: str) -> str:
        if not isinstance(code, str): return ""
        return code.replace("```", "").strip()

    def _tokenize(self, text):
        return self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        code = self._clean_code(str(self.data.at[idx, "code"]))
        if len(code) < 5: code = "print('error')"
        label = int(self.data.at[idx, "label"])
        
        lang_str = str(self.data.at[idx, "language"]).lower().strip()
        lang_id = self.language_map.get(lang_str, self.language_map.get('unknown', 0))

        enc = self._tokenize(code)
        stylo = self.stylo_extractor.extract(code) if self.stylo_extractor else np.zeros(13, dtype=np.float32)

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "stylo_feats": torch.tensor(stylo, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long)
        }

        # Gestione Augmentation (Consistency Loss)
        if self.has_paired_data and pd.notna(self.data.at[idx, "aug_code"]):
            aug_code = self._clean_code(self.data.at[idx, "aug_code"])
            aug_enc = self._tokenize(aug_code)
            aug_stylo = self.stylo_extractor.extract(aug_code) if self.stylo_extractor else np.zeros(13, dtype=np.float32)
            item["input_ids_aug"] = aug_enc["input_ids"].squeeze(0)
            item["attention_mask_aug"] = aug_enc["attention_mask"].squeeze(0)
            item["stylo_feats_aug"] = torch.tensor(aug_stylo, dtype=torch.float32)
            item["has_aug"] = torch.tensor(1, dtype=torch.long)
        else:
            item["input_ids_aug"] = torch.zeros_like(item["input_ids"])
            item["attention_mask_aug"] = torch.zeros_like(item["attention_mask"])
            item["stylo_feats_aug"] = torch.zeros_like(item["stylo_feats"])
            item["has_aug"] = torch.tensor(0, dtype=torch.long)
        
        return item

# -----------------------------------------------------------------------------
# 3. Data Loading Logic
# -----------------------------------------------------------------------------
def load_and_preprocess(file_path: str, max_code_chars: int = 20000) -> pd.DataFrame:
    if not os.path.exists(file_path): return pd.DataFrame()
    df = pd.read_parquet(file_path)
    
    rename_map = {}
    if 'original_code' in df.columns: rename_map['original_code'] = 'code'
    if 'augmented_code' in df.columns: rename_map['augmented_code'] = 'aug_code'
    if 'original_lang' in df.columns: rename_map['original_lang'] = 'language'
    if rename_map: df = df.rename(columns=rename_map)

    df['language'] = df['language'].astype(str).str.lower().str.strip()
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
    df = df.dropna(subset=['code', 'label'])
    df = df[df['code'].str.strip().str.len() > 10] 
    df['code'] = df['code'].str.slice(0, max_code_chars)
    return df.reset_index(drop=True)

def smart_balance_dataset(df: pd.DataFrame, max_python_samples: int = 30000) -> pd.DataFrame:
    if df.empty: return df
    
    df_python = df[df['language'] == 'python']
    df_others = df[df['language'] != 'python']
    
    logger.info(f"Balancing: Python={len(df_python)} | Others (Go, C#, etc.)={len(df_others)}")
    
    if len(df_python) > max_python_samples:
        df_python = df_python.sample(n=max_python_samples, random_state=42)
        
    return pd.concat([df_python, df_others]).sample(frac=1, random_state=42).reset_index(drop=True)

def load_data(config: dict, tokenizer):
    logger.info(">>> LOADING DATASETS WITH 'DATA FLIPPING' <<<")
    
    # 1. Carica Train Base
    train_df = load_and_preprocess(config["data"]["train_path"])
    
    # 2. Carica Augmented Data
    aug_path = os.path.join(os.path.dirname(config["data"]["train_path"]), "train_augmented.parquet")
    
    if os.path.exists(aug_path):
        logger.info(f"Processing Augmented Data from {aug_path}")
        aug_raw = pd.read_parquet(aug_path)
        
        # Rename di sicurezza
        if 'original_code' in aug_raw.columns: 
            aug_raw = aug_raw.rename(columns={'original_code': 'code', 'augmented_code': 'aug_code', 'original_lang': 'language'})


        df_consistency = aug_raw[['code', 'label', 'language', 'aug_code']].copy()
        df_new_langs = aug_raw.copy()
        
        df_new_langs['code'] = aug_raw['aug_code']
        df_new_langs['language'] = aug_raw['aug_lang']
        df_new_langs['aug_code'] = aug_raw['code']
        
        df_new_langs = df_new_langs[['code', 'label', 'language', 'aug_code']]
        df_new_langs = df_new_langs.dropna(subset=['code'])
        df_new_langs = df_new_langs[df_new_langs['code'].str.len() > 10]
        
        logger.info(f"Injecting {len(df_new_langs)} samples of NEW languages (Go, C#, PHP...) into training!")
        
        # 3. Unione Totale: Train Originale + Consistency + Nuovi Linguaggi
        train_df = pd.concat([train_df, df_consistency, df_new_langs], ignore_index=True)
        
        train_df = train_df.drop_duplicates(subset=['code', 'language']).reset_index(drop=True)

    else:
        train_df['aug_code'] = None

    # 4. Bilanciamento
    if config["data"].get("balance_languages", True):
        py_cap = config["data"].get("samples_per_lang", 30000)
        train_df = smart_balance_dataset(train_df, max_python_samples=py_cap)

    logger.info(f"FINAL TRAIN SIZE: {len(train_df)}")
    logger.info(f"FINAL LANGUAGE DISTRIBUTION:\n{train_df['language'].value_counts()}")

    # 5. Validation
    val_df = load_and_preprocess(config["data"]["val_path"])
    VAL_SUBSET = 4000
    if len(val_df) > VAL_SUBSET:
        try:
            val_df = val_df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), VAL_SUBSET // 2), random_state=42)
            ).reset_index(drop=True)
        except:
            val_df = val_df.sample(VAL_SUBSET, random_state=42).reset_index(drop=True)

    # 6. Mappa Lingue
    all_langs = pd.concat([train_df['language'], val_df['language']]).unique()
    lang_map = {l: i for i, l in enumerate(sorted([str(l) for l in all_langs]))}
    if 'unknown' not in lang_map: lang_map['unknown'] = len(lang_map)
    
    logger.info(f"Detected Languages Map: {lang_map}")
    
    train_ds = CodeDataset(train_df, tokenizer, lang_map, config["data"]["max_length"], augment=True)
    val_ds = CodeDataset(val_df, tokenizer, lang_map, config["data"]["max_length"], augment=False)
    
    train_sampler = BalancedBatchSampler(train_ds.labels_list, batch_size=config["training"]["batch_size"])
    
    return train_ds, val_ds, train_sampler, train_df