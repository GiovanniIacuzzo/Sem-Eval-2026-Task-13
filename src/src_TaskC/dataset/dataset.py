import os
import random
import logging
import pandas as pd
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------------------------------------------------------
# Logger Setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# -----------------------------------------------------------------------------
# PyTorch Dataset Class
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
    """
    Optimized PyTorch Dataset per Task C.
    Mantiene il rumore strutturale (spazi/newlines) ma RIMUOVE il masking dei token
    per preservare gli artefatti sintattici specifici delle AI.
    """
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 tokenizer, 
                 language_map: Dict[str, int],
                 max_length: int = 512, 
                 augment: bool = False):
        
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.language_map = language_map
        self.max_length = max_length
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def _structural_noise(self, code: str) -> str:
        """
        Simula variazioni di stile umano (spaziature, righe vuote)
        senza distruggere la sintassi del codice.
        """
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            # Aggiunge righe vuote casuali
            if random.random() < 0.05:
                new_lines.append("")
            
            # Aggiunge spazi casuali a fine riga (trailing spaces)
            if random.random() < 0.05:
                line = line + " " * random.randint(1, 3)
            
            # Aggiunge un rientro errato occasionale
            if random.random() < 0.01:
                line = " " + line

            new_lines.append(line)
        
        return "\n".join(new_lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        code = str(row["code"])
        label = int(row["label"]) 
        lang_str = str(row["language"]).lower()
        
        lang_id = self.language_map.get(lang_str, -1)
        
        # 1. Augmentation Strutturale (Safe)
        if self.augment:
            code = self._structural_noise(code)

        # 2. Random Cropping (Se il codice è lunghissimo)
        # Prende una finestra casuale invece di troncare sempre la fine
        if self.augment and len(code) > self.max_length * 4:
            max_start = len(code) - int(self.max_length * 3.5)
            if max_start > 0:
                start_idx = random.randint(0, max_start)
                code = code[start_idx : start_idx + int(self.max_length * 4)]

        # 3. Tokenization standard
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # --- RIMOSSO IL MASKING MLM ---
        # Il masking casuale (BERT style) danneggia la capacità del modello
        # di rilevare pattern sottili generati dalle LLM.

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long)
        }

# -----------------------------------------------------------------------------
# Data Processing Helpers
# -----------------------------------------------------------------------------
def load_and_preprocess(file_path: str, max_code_chars: int = 20000) -> pd.DataFrame:
    columns = ['code', 'label', 'language'] 
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        df = pd.read_parquet(file_path, columns=columns)
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise e
        
    df['language'] = df['language'].str.lower()
    df = df.dropna(subset=['code']).reset_index(drop=True)
    # Filtro codici troppo brevi (spesso rumore)
    df = df[df['code'].str.len() > 15].copy() 
    df['code'] = df['code'].str.slice(0, max_code_chars)
    return df

def balance_languages(df: pd.DataFrame, target_samples: int = 3000) -> pd.DataFrame:
    """
    Limita il numero massimo di sample per linguaggio per evitare che
    linguaggi popolari (Python/Java) dominino il training.
    """
    logger.info(f"Balancing languages... Max cap per lang: {target_samples}")
    df_list = []
    for lang, count in df['language'].value_counts().items():
        group = df[df['language'] == lang]
        if count > target_samples:
            df_list.append(group.sample(n=target_samples, random_state=42))
        else:
            df_list.append(group)
    
    balanced_df = pd.concat(df_list).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df

def get_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Calcola i pesi inversi per bilanciare la Loss Function.
    Cruciale per Task C dove Hybrid e Adversarial sono rari.
    """
    labels = df['label'].values
    classes = np.unique(labels)
    
    # 'balanced': n_samples / (n_classes * np.bincount(y))
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    
    # Conversione in tensore
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"Computed Class Weights: {weight_tensor}")
    return weight_tensor

# -----------------------------------------------------------------------------
# Main Data Loading Interface
# -----------------------------------------------------------------------------
def load_data(config: dict, tokenizer, device=torch.device("cpu")) -> Tuple[CodeDataset, CodeDataset, torch.Tensor]:
    logger.info(">>> Loading Data for Task C (Multiclass) <<<")
    
    train_path = config["data"]["train_path"]
    val_path = config["data"]["val_path"]
    
    train_df = load_and_preprocess(train_path)
    val_df   = load_and_preprocess(val_path)
    
    # 1. Bilanciamento per Linguaggio (evita overfitting su Python)
    samples_cap = config["data"].get("samples_per_lang", 4000)
    if config["data"].get("balance_languages", True):
        train_df = balance_languages(train_df, target_samples=samples_cap)
    
    # 2. Calcolo dei pesi per la Loss Function (gestisce sbilanciamento label)
    class_weights = get_class_weights(train_df, device)
    
    # DANN Mapping
    target_langs = config["model"].get("languages", ["python", "java", "cpp"])
    language_map = {lang.lower(): i for i, lang in enumerate(target_langs)}
    
    max_len = config["data"].get("max_length", 512)
    
    train_dataset = CodeDataset(train_df, tokenizer, language_map, max_length=max_len, augment=True)
    val_dataset = CodeDataset(val_df, tokenizer, language_map, max_length=max_len, augment=False)

    logger.info(f"Dataset Stats: Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    logger.info(f"Train Label Distribution:\n{train_df['label'].value_counts().sort_index()}")
    
    # Restituiamo i dataset E i pesi per la loss
    return train_dataset, val_dataset, class_weights