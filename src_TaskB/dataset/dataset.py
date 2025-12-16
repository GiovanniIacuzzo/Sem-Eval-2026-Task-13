import random
import logging
import gc
import pandas as pd
import numpy as np
import torch
from typing import Tuple, Dict, List
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

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
# 1. MAPPINGS & CONFIGURATION
# -----------------------------------------------------------------------------
# ID univoci per le 31 classi
GENERATOR_MAP = {
    '01-ai/yi-coder-1.5b': 0, '01-ai/yi-coder-1.5b-chat': 1,
    'bigcode/starcoder': 2, 'bigcode/starcoder2-15b': 3, 'bigcode/starcoder2-3b': 4,
    'deepseek-ai/deepseek-coder-1.3b-base': 5, 'deepseek-ai/deepseek-r1': 6, 'deepseek-ai/deepseek-v3-0324': 7,
    'gemma-3-27b-it': 8, 'gemma-3n-e4b-it': 9, 'google/codegemma-2b': 10,
    'gpt-4o': 11,
    'human': 12,
    'ibm-granite/granite-3.2-2b-instruct': 13, 'ibm-granite/granite-3.3-8b-base': 14, 'ibm-granite/granite-3.3-8b-instruct': 15,
    'meta-llama/llama-3.1-8b': 16, 'meta-llama/llama-3.1-8b-instruct': 17, 'meta-llama/llama-3.2-11b-vision-instruct': 18,
    'meta-llama/llama-3.2-1b': 19, 'meta-llama/llama-3.2-3b': 20,
    'microsoft/phi-3-medium-4k-instruct': 21, 'microsoft/phi-3-mini-4k-instruct': 22, 'microsoft/phi-3-small-8k-instruct': 23,
    'mistralai/devstral-small-2505': 24, 'mistralai/mistral-7b-instruct-v0.3': 25,
    'qwen/qwen2.5-72b-instruct': 26, 'qwen/qwen2.5-codder-14b-instruct': 27, 'qwen/qwen2.5-coder-1.5b': 28,
    'qwen/qwen2.5-coder-1.5b-instruct': 29, 'qwen/qwq-32b': 30,
}

# Raggruppamento per famiglia (Essenziale per validazione "Unseen")
FAMILY_MAP = {
    'yi': ['01-ai/yi-coder-1.5b', '01-ai/yi-coder-1.5b-chat'],
    'bigcode': ['bigcode/starcoder', 'bigcode/starcoder2-15b', 'bigcode/starcoder2-3b'],
    'deepseek': ['deepseek-ai/deepseek-coder-1.3b-base', 'deepseek-ai/deepseek-r1', 'deepseek-ai/deepseek-v3-0324'],
    'gemma': ['gemma-3-27b-it', 'gemma-3n-e4b-it', 'google/codegemma-2b'],
    'gpt': ['gpt-4o'],
    'human': ['human'],
    'granite': ['ibm-granite/granite-3.2-2b-instruct', 'ibm-granite/granite-3.3-8b-base', 'ibm-granite/granite-3.3-8b-instruct'],
    'llama': ['meta-llama/llama-3.1-8b', 'meta-llama/llama-3.1-8b-instruct', 'meta-llama/llama-3.2-11b-vision-instruct', 'meta-llama/llama-3.2-1b', 'meta-llama/llama-3.2-3b'],
    'phi': ['microsoft/phi-3-medium-4k-instruct', 'microsoft/phi-3-mini-4k-instruct', 'microsoft/phi-3-small-8k-instruct'],
    'mistral': ['mistralai/devstral-small-2505', 'mistralai/mistral-7b-instruct-v0.3'],
    'qwen': ['qwen/qwen2.5-72b-instruct', 'qwen/qwen2.5-codder-14b-instruct', 'qwen/qwen2.5-coder-1.5b', 'qwen/qwen2.5-coder-1.5b-instruct', 'qwen/qwq-32b']
}

# -----------------------------------------------------------------------------
# 2. DATA AUGMENTATION LOGIC
# -----------------------------------------------------------------------------
class CodeAugmenter:
    """
    Augmentation specifica per il codice.
    Non altera la semantica gravemente, ma cambia la formattazione per evitare overfitting.
    """
    @staticmethod
    def whitespace_noise(code: str, prob: float = 0.3) -> str:
        """Inserisce o rimuove a capo/spazi casualmente."""
        if random.random() > prob:
            return code
            
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            # 10% chance di aggiungere una riga vuota
            if random.random() < 0.1:
                new_lines.append("") 
            new_lines.append(line)
            # 5% chance di aggiungere spazi alla fine
            if random.random() < 0.05:
                new_lines[-1] += " " * random.randint(1, 4)
                
        return '\n'.join(new_lines)

# -----------------------------------------------------------------------------
# 3. DATASET CLASS
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        tokenizer, 
        max_length: int = 512, 
        mode: str = "train", 
        overlap: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.overlap = overlap
        self.mask_token_id = tokenizer.mask_token_id
        
        self.samples = []
        # Reset index per evitare bug con iloc
        self.data = dataframe.reset_index(drop=True)
        
        logger.info(f"Initializing Dataset ({mode}). Rows: {len(self.data)}")
        
        if self.mode == "train":
            # Sliding Window per il training: più dati, copertura totale
            self._prepare_sliding_window()
        else:
            # Validation/Test: 1 campione = 1 entry (con Head+Tail truncation)
            self.samples = [(i, 0, -1) for i in range(len(self.data))]

    def _prepare_sliding_window(self):
        """Pre-calcola gli indici per lo sliding window."""
        stride_tokens = int(self.max_length * 0.75) # 25% overlap
        # Stima conservativa chars -> tokens (1 token ~= 3.5 chars)
        stride_chars = int(stride_tokens * 3.5)
        window_chars = int(self.max_length * 3.5)

        for idx, row in self.data.iterrows():
            code_len = len(row['code'])
            if code_len <= window_chars:
                self.samples.append((idx, 0, -1))
            else:
                for start in range(0, code_len, stride_chars):
                    # Evita chunk troppo piccoli alla fine (< 50 chars)
                    if start + 50 < code_len:
                        self.samples.append((idx, start, start + window_chars))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row_idx, start, end = self.samples[idx]
        row = self.data.iloc[row_idx]
        
        code = str(row["code"])
        label = int(row["label"])

        # 1. Slicing Strategy
        if self.mode == "train":
            # Sliding Window
            text_chunk = code[start:end] if end != -1 else code
            # Augmentation
            text_chunk = CodeAugmenter.whitespace_noise(text_chunk)
        else:
            # Head + Tail Truncation per Validation
            # Prende l'inizio (imports, definizioni) e la fine (return, main)
            limit = int(self.max_length * 3.5)
            if len(code) > limit:
                split_point = limit // 2
                text_chunk = code[:split_point] + "\n" + code[-split_point:]
            else:
                text_chunk = code

        # 2. Tokenization
        encoding = self.tokenizer(
            text_chunk,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # 3. Augmentation: Token Masking (MLM style)
        # Rende il modello robusto a cambiamenti di nome variabili
        if self.mode == "train" and random.random() < 0.15:
            probability_matrix = torch.full(input_ids.shape, 0.15)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                input_ids.tolist(), already_has_special_tokens=True
            )
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            probability_matrix.masked_fill_(attention_mask == 0, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            input_ids[masked_indices] = self.mask_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }

# -----------------------------------------------------------------------------
# 4. PREPROCESSING & SPLITTING STRATEGIES
# -----------------------------------------------------------------------------
def load_base_dataframe(file_path: str, task_type: str = "multiclass") -> pd.DataFrame:
    """Carica e pulisce il dataframe base."""
    columns = ['code', 'language', 'generator']
    if task_type == "binary" and 'label' in pd.read_parquet(file_path).columns:
        columns.append('label')

    df = pd.read_parquet(file_path, columns=columns)
    df = df.dropna(subset=['code']).reset_index(drop=True)
    df = df[df['code'].str.len() > 20].copy() # Rimuove snippet inutilizzabili
    
    # Normalizzazione
    df['generator'] = df['generator'].str.lower()
    df['language'] = df['language'].str.lower()

    if task_type == "multiclass":
        # Filtra solo generatori noti
        known_gens = set(GENERATOR_MAP.keys())
        df = df[df['generator'].isin(known_gens)].copy()
        df['label'] = df['generator'].map(GENERATOR_MAP).astype(int)
    
    return df

def create_unseen_author_split(df: pd.DataFrame, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    STRATEGIA VINCENTE: Simula 'Unseen Authors'.
    Invece di splittare a caso, riserva interi modelli per la validazione.
    """
    logger.info("Creating 'Unseen Author' Validation Split...")
    val_indices = []
    
    # Per ogni famiglia, cerchiamo di mettere 1 modello interamente in validation
    for family, models in FAMILY_MAP.items():
        available_models = [m for m in models if m in df['generator'].unique()]
        
        if len(available_models) > 1:
            # Se la famiglia ha più modelli, ne prendiamo uno a caso come 'unseen'
            # (tranne human che va splittato randomicamente)
            if family == 'human':
                human_indices = df[df['generator'] == 'human'].index
                # Prendi un 10% random di umani
                val_subset = np.random.choice(human_indices, size=int(len(human_indices) * val_size), replace=False)
                val_indices.extend(val_subset)
            else:
                # Scegli un modello da escludere dal training (es. Llama-3.2-1b)
                # Preferiamo modelli con meno dati per non impoverire troppo il train
                models_sorted_by_count = df[df['generator'].isin(available_models)]['generator'].value_counts().index.tolist()
                # Prendiamo il secondo o l'ultimo (quello con meno dati o intermedio)
                held_out_model = models_sorted_by_count[-1] 
                
                logger.info(f"Holding out model for validation: {held_out_model} (Family: {family})")
                val_indices.extend(df[df['generator'] == held_out_model].index)
        
        elif len(available_models) == 1:
            # Se c'è un solo modello (es. GPT-4o), dobbiamo fare split random classico
            single_model_indices = df[df['generator'] == available_models[0]].index
            val_subset = np.random.choice(single_model_indices, size=int(len(single_model_indices) * val_size), replace=False)
            val_indices.extend(val_subset)

    val_df = df.loc[val_indices].copy()
    train_df = df.drop(val_indices).copy()
    
    # Shuffle
    return train_df.sample(frac=1, random_state=42), val_df.sample(frac=1, random_state=42)

def balance_languages(df: pd.DataFrame, max_samples: int = 5000) -> pd.DataFrame:
    """Evita che Python domini il dataset."""
    df_list = []
    # Raggruppa per Label (Modello) E Linguaggio
    # Così garantiamo che anche modelli rari in linguaggi rari siano preservati
    df['strat_key'] = df['generator'] + "_" + df['language']
    
    for _, group in df.groupby('strat_key'):
        if len(group) > max_samples:
            df_list.append(group.sample(n=max_samples, random_state=42))
        else:
            df_list.append(group)
            
    balanced = pd.concat(df_list).sample(frac=1, random_state=42).reset_index(drop=True)
    balanced = balanced.drop(columns=['strat_key'])
    return balanced

# -----------------------------------------------------------------------------
# MAIN INTERFACE
# -----------------------------------------------------------------------------
def load_data(config: dict, tokenizer) -> Tuple[CodeDataset, CodeDataset, pd.DataFrame, pd.DataFrame]:
    """
    Carica i dati e prepara i Dataset PyTorch.
    """
    logger.info(">>> Starting Data Loading Pipeline <<<")
    
    # 1. Load Raw Data
    # Carichiamo sia train che validation originali, e li uniamo per rifare lo split noi
    df_train_raw = load_base_dataframe(config["data"]["train_path"])
    
    # Opzionale: se hai un validation.parquet fornito, lo uniamo al train per poi risplittare
    # intelligentemente, OPPURE lo usiamo come test set aggiuntivo.
    # Per sicurezza, uniamo tutto e rifacciamo lo split per garantire la logica "Unseen".
    if "val_path" in config["data"] and config["data"]["val_path"]:
        try:
            df_val_raw = load_base_dataframe(config["data"]["val_path"])
            full_df = pd.concat([df_train_raw, df_val_raw], ignore_index=True)
        except:
            full_df = df_train_raw
    else:
        full_df = df_train_raw

    # 2. Demo Mode
    if config.get("demo", {}).get("active", False):
        logger.warning("!!! DEMO MODE ACTIVE - Using tiny subset !!!")
        full_df = full_df.sample(n=2000, random_state=42)

    # 3. Balancing (Opzionale ma raccomandato per T4 per non sprecare epoche su Python)
    if config["data"].get("balance_languages", True):
        full_df = balance_languages(full_df, max_samples=4000)

    # 4. Strategic Splitting
    # Qui creiamo il vero set di addestramento e validazione
    train_df, val_df = create_unseen_author_split(full_df, val_size=0.15)
    
    logger.info(f"Train Rows: {len(train_df)} | Val Rows: {len(val_df)}")
    logger.info(f"Train Classes: {train_df['label'].nunique()} | Val Classes: {val_df['label'].nunique()}")

    # 5. Dataset Creation
    train_dataset = CodeDataset(
        train_df, 
        tokenizer, 
        max_length=config["data"].get("max_length", 512), 
        mode="train"
    )
    
    val_dataset = CodeDataset(
        val_df, 
        tokenizer, 
        max_length=config["data"].get("max_length", 512), 
        mode="val"
    )

    return train_dataset, val_dataset, train_df, val_df