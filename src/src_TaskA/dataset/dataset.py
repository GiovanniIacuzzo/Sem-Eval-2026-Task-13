import re
import os
import random
import logging
import pandas as pd
import torch
import numpy as np
from typing import Dict, List, Iterator
from torch.utils.data import Dataset, Sampler

# -----------------------------------------------------------------------------
# FIX IMPORTS & FALLBACK
# -----------------------------------------------------------------------------
try:
    from src.src_TaskA.features.stylometry import StylometryExtractor
except ImportError:
    try:
        from src_TaskA.features.stylometry import StylometryExtractor
    except ImportError:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("WARNING: StylometryExtractor NOT FOUND. Using DUMMY (Zeros).")
        print("Check your folder structure or imports in dataset.py")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        class StylometryExtractor:
            def extract(self, code): 
                # Deve essere 13 per matchare il config
                return np.zeros(13, dtype=np.float32)

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
        self.n_batches = int(len(self.labels) / self.batch_size)
        self.n_samples_per_class = self.batch_size // 2

    def __iter__(self) -> Iterator[int]:
        np.random.shuffle(self.cls0_indices)
        np.random.shuffle(self.cls1_indices)
        ptr0 = 0
        ptr1 = 0
        for _ in range(self.n_batches):
            batch_indices = []
            # Human
            end0 = ptr0 + self.n_samples_per_class
            if end0 > self.n_cls0:
                np.random.shuffle(self.cls0_indices)
                ptr0 = 0
                end0 = self.n_samples_per_class
            batch_indices.extend(self.cls0_indices[ptr0:end0])
            ptr0 = end0
            # AI
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
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 tokenizer, 
                 language_map: Dict[str, int],
                 max_length: int = 512, 
                 augment: bool = False):
        
        self.data = dataframe.reset_index(drop=True)
        self.labels_list = self.data['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.language_map = language_map
        self.max_length = max_length
        self.augment = augment
        
        self.stylo_extractor = StylometryExtractor()

        # Verifica presenza colonna per i dati paired
        self.has_paired_data = 'aug_code' in self.data.columns

    def __len__(self) -> int:
        return len(self.data)

    def _clean_code(self, code: str) -> str:
        if not isinstance(code, str): return ""
        code = code.strip()
        code = re.sub(r'\n{3,}', '\n\n', code)
        return code

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # --- A. Dati Principali ---
        code_raw = str(self.data.at[idx, "code"])
        label = int(self.data.at[idx, "label"]) 
        lang_str = str(self.data.at[idx, "language"]).lower()
        lang_id = self.language_map.get(lang_str, 0)
        
        code = self._clean_code(code_raw)
        if len(code) < 5: code = "print('error')"

        # Tokenizzazione Standard
        encoding = self._tokenize(code)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Stilometria Standard
        stylo_feats = self.stylo_extractor.extract(code)
        stylo_tensor = torch.tensor(stylo_feats, dtype=torch.float32)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "stylo_feats": stylo_tensor,
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long)
        }

        # --- B. Dati Paired (Consistency) ---
        if self.has_paired_data:
            aug_code_raw = self.data.at[idx, "aug_code"]
            
            # Controlla se aug_code è valido
            if isinstance(aug_code_raw, str) and len(aug_code_raw) > 5:
                aug_code = self._clean_code(aug_code_raw)
                
                # Tokenizzazione Augmented
                aug_enc = self._tokenize(aug_code)
                
                # Stilometria Augmented
                aug_stylo = self.stylo_extractor.extract(aug_code)
                
                item["input_ids_aug"] = aug_enc["input_ids"].squeeze(0)
                item["attention_mask_aug"] = aug_enc["attention_mask"].squeeze(0)
                item["stylo_feats_aug"] = torch.tensor(aug_stylo, dtype=torch.float32)
                item["has_aug"] = torch.tensor(1, dtype=torch.long)
            else:
                item["input_ids_aug"] = torch.zeros_like(input_ids)
                item["attention_mask_aug"] = torch.zeros_like(attention_mask)
                item["stylo_feats_aug"] = torch.zeros_like(stylo_tensor)
                item["has_aug"] = torch.tensor(0, dtype=torch.long)
        
        return item

# -----------------------------------------------------------------------------
# 3. Data Loading Utils (CON FILTRO RILASSATO)
# -----------------------------------------------------------------------------
def load_and_preprocess(file_path: str, max_code_chars: int = 20000) -> pd.DataFrame:
    if not os.path.exists(file_path):
        return pd.DataFrame()

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise e
    
    # Rinomina per compatibilità
    if 'original_code' in df.columns and 'augmented_code' in df.columns:
        logger.info("Detected Paired Dataset format. Renaming columns...")
        df = df.rename(columns={
            'original_code': 'code',
            'augmented_code': 'aug_code',
            'original_lang': 'language'
        })
    
    # Standard check
    required = ['code', 'label', 'language']
    if not all(c in df.columns for c in required):
        logger.warning(f"File {file_path} missing required columns. Found: {df.columns}")
        return pd.DataFrame()

    # Normalizzazione Base
    df['language'] = df['language'].str.lower().str.strip()
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
    
    initial_len = len(df)
    
    # [FIX] FILTRAGGIO MENO AGGRESSIVO
    # 1. Rimuovi righe vuote o NaN
    df = df.dropna(subset=['code', 'label'])
    
    # 2. Rilassiamo il limite di lunghezza (10 char bastano per "print(x)")
    df = df[df['code'].str.strip().str.len() > 10]
    
    # 3. RIMOSSO: Filtro Regex su simboli specifici
    # code_indicators = ['(', ')', '=', '{', '}', 'def ', 'import ', 'class ', ';', 'return', 'var ', 'func ']
    # pattern = '|'.join([re.escape(x) for x in code_indicators])
    # df = df[df['code'].str.contains(pattern, regex=True)]
    
    # 4. Rimuovi specifici "errori" noti (Testo boilerplate AI)
    df = df[~df['code'].str.startswith("valid version for the language")]
    
    # Troncatura finale
    df['code'] = df['code'].str.slice(0, max_code_chars)
    
    filtered_len = len(df)
    if initial_len - filtered_len > 0:
        logger.info(f"Data Cleaning on {os.path.basename(file_path)}: Dropped {initial_len - filtered_len} rows (Relaxed Filter).")
    
    return df.reset_index(drop=True)

def balance_languages(df: pd.DataFrame, target_samples: int = 5000) -> pd.DataFrame:
    if df.empty or 'language' not in df.columns: return df
    
    logger.info("Balancing languages...")
    df_list = []
    stats = df['language'].value_counts()
    
    for lang, count in stats.items():
        group = df[df['language'] == lang]
        if count > target_samples:
            df_list.append(group.sample(n=target_samples, random_state=42))
        else:
            df_list.append(group)
            
    if not df_list: return df
    return pd.concat(df_list).sample(frac=1, random_state=42).reset_index(drop=True)

def load_data(config: dict, tokenizer):
    logger.info(">>> Loading Data with Consistency Support <<<")
    
    train_path = config["data"]["train_path"]
    val_path = config["data"]["val_path"]
    
    # 1. Train Originale (Verrà pulito automaticamente)
    train_df = load_and_preprocess(train_path)
    
    # 2. Train Paired (Già pulito, ma passa comunque il check)
    train_dir = os.path.dirname(train_path)
    paired_path = os.path.join(train_dir, "train_augmented.parquet") 
    
    if os.path.exists(paired_path):
        logger.info(f"FOUND PAIRED DATA: {paired_path}")
        paired_df = load_and_preprocess(paired_path)
        
        # Merge: train originale + dati paired
        train_df = pd.concat([train_df, paired_df], ignore_index=True)
        logger.info(f"Merged paired data. Total Train Size: {len(train_df)}")
        logger.info(f"Paired samples count: {train_df['aug_code'].notna().sum()}")
    else:
        logger.warning(f"No paired data found at {paired_path}. Consistency Loss will be disabled.")
        train_df['aug_code'] = None

    # 3. Validation
    val_df = load_and_preprocess(val_path)
    
    # 4. Bilanciamento
    samples_cap = config["data"].get("samples_per_lang", 6000)
    if config["data"].get("balance_languages", True):
        train_df = balance_languages(train_df, target_samples=samples_cap)
    
    # 5. Mappa Lingue
    unique_langs = sorted(list(set(train_df['language'].unique()) | set(val_df['language'].unique())))
    language_map = {lang: i for i, lang in enumerate(unique_langs)}
    if 'unknown' not in language_map: language_map['unknown'] = len(language_map)
    
    # 6. Dataset
    max_len = config["data"].get("max_length", 512)
    train_dataset = CodeDataset(train_df, tokenizer, language_map, max_length=max_len, augment=True)
    val_dataset = CodeDataset(val_df, tokenizer, language_map, max_length=max_len, augment=False)
    
    # 7. Sampler
    batch_size = config["training"]["batch_size"]
    train_sampler = BalancedBatchSampler(train_dataset.labels_list, batch_size=batch_size)
    
    return train_dataset, val_dataset, train_sampler, train_df