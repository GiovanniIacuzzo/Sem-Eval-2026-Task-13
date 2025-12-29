import os
import logging
import pandas as pd
import torch
import numpy as np
from typing import Dict, List, Iterator
from torch.utils.data import Dataset, Sampler
from src.src_TaskA.features.stylometry import StylometryExtractor

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
        
        if self.n_cls0 > 0 and self.n_cls1 > 0:
            min_size = min(self.n_cls0, self.n_cls1)
            self.n_batches = int((len(self.labels)) / self.batch_size)
        else:
            self.n_batches = int(len(self.labels) / self.batch_size)
            
        self.n_samples_per_class = self.batch_size // 2

    def __iter__(self) -> Iterator[int]:
        np.random.shuffle(self.cls0_indices)
        np.random.shuffle(self.cls1_indices)
        
        ptr0 = 0
        ptr1 = 0
        
        from itertools import cycle
        iter0 = cycle(self.cls0_indices)
        iter1 = cycle(self.cls1_indices)

        for _ in range(self.n_batches):
            batch_indices = []
            
            for _ in range(self.n_samples_per_class):
                batch_indices.append(next(iter0))
                
            for _ in range(self.batch_size - self.n_samples_per_class):
                batch_indices.append(next(iter1))

            np.random.shuffle(batch_indices)
            for idx in batch_indices:
                yield int(idx)

    def __len__(self) -> int:
        return self.n_batches * self.batch_size

# -----------------------------------------------------------------------------
# 2. Dataset Class
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, language_map: Dict[str, int], max_length: int = 512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.language_map = language_map
        self.max_length = max_length
        
        # Inizializza estrattore stilometrico
        self.stylo_extractor = StylometryExtractor()

        # Lista labels per il sampler
        self.labels_list = self.data['label'].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.data)

    def _clean_code(self, code: str) -> str:
        if not isinstance(code, str): return ""
        # Rimuove backticks markdown se presenti
        return code.replace("```python", "").replace("```java", "").replace("```cpp", "").replace("```", "").strip()

    def _tokenize(self, text):
        return self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1. Recupero Dati Grezzi
        raw_code = str(self.data.at[idx, "code"])
        code = self._clean_code(raw_code)
        
        # Fallback per codici vuoti/corrotti
        if len(code) < 5: 
            code = "print('error_empty_code')"
            
        label = int(self.data.at[idx, "label"])
        
        # 2. Tokenizzazione (Semantica)
        enc = self._tokenize(code)
        
        # 3. Estrazione Features (Stilometria)
        try:
            stylo = self.stylo_extractor.extract(code)
        except Exception as e:
            logger.warning(f"Stylo error at idx {idx}: {e}")
            stylo = np.zeros(15, dtype=np.float32)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "stylo_feats": torch.tensor(stylo, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# -----------------------------------------------------------------------------
# 3. Data Loading & Cleaning Logic
# -----------------------------------------------------------------------------
def load_and_preprocess(file_path: str, max_code_chars: int = 20000) -> pd.DataFrame:
    """Carica parquet e fa pulizia base."""
    if not os.path.exists(file_path): 
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame()
        
    df = pd.read_parquet(file_path)
    
    # Standardizza nomi colonne
    rename_map = {}
    if 'original_code' in df.columns: rename_map['original_code'] = 'code'
    if 'original_lang' in df.columns: rename_map['original_lang'] = 'language'
    if rename_map: df = df.rename(columns=rename_map)

    # Pulizia Base
    df['language'] = df['language'].astype(str).str.lower().str.strip()
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
    
    # Rimuovi righe non valide
    initial_len = len(df)
    df = df.dropna(subset=['code', 'label'])
    df = df[df['label'].isin([0, 1])]
    df = df[df['code'].str.strip().str.len() > 10]
    
    # Tronca codici troppo lunghi
    df['code'] = df['code'].str.slice(0, max_code_chars)
    
    logger.info(f"Loaded {file_path}: {len(df)} samples (dropped {initial_len - len(df)})")
    return df.reset_index(drop=True)

def downsample_dominant_language(df: pd.DataFrame, target_lang: str = 'python', max_samples: int = 35000) -> pd.DataFrame:
    """
    Se Python è > 80% del dataset, lo riduciamo a 'max_samples' per evitare overfitting
    sulla sintassi Python, mantenendo intatti gli altri linguaggi rari.
    """
    if df.empty: return df
    
    df_dominant = df[df['language'] == target_lang]
    df_others = df[df['language'] != target_lang]
    
    if len(df_dominant) > max_samples:
        logger.info(f"Downsampling {target_lang} from {len(df_dominant)} to {max_samples}...")
        df_dominant = df_dominant.sample(n=max_samples, random_state=42)
    
    # Uniamo e mescoliamo
    balanced_df = pd.concat([df_dominant, df_others]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"New Balance: {target_lang}={len(df_dominant)} | Others={len(df_others)}")
    return balanced_df

def load_full_data_for_kfold(config):
    """
    Funzione principale per caricare TUTTI i dati (Train + Val originali).
    Non carica più augmentation esterne.
    """
    logger.info(">>> LOADING FULL DATASET (Clean Strategy) <<<")
    
    # 1. Load Train & Val originali
    train_path = config["data"]["train_path"]
    val_path = config["data"]["val_path"]
    
    df_train = load_and_preprocess(train_path)
    df_val = load_and_preprocess(val_path)
    
    # 2. Merge in un unico dataset
    full_df = pd.concat([df_train, df_val]).reset_index(drop=True)
    logger.info(f"Merged Total Size: {len(full_df)}")
    
    # 3. Gestione Dominanza Linguaggio
    python_cap = config["data"].get("max_python_samples", 40000)
    full_df = downsample_dominant_language(full_df, target_lang='python', max_samples=python_cap)
    
    # 4. Report Finale Distribuzione
    logger.info("Final Language Distribution:")
    logger.info(full_df['language'].value_counts())
    
    return full_df