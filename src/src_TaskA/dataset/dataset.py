import re
import random
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
# 1. Custom Sampler per SupCon
# -----------------------------------------------------------------------------
class BalancedBatchSampler(Sampler):
    """
    Garantisce che ogni batch abbia un mix equilibrato di classi (es. 50% classe 0, 50% classe 1).
    Cruciale per la Contrastive Loss (SupCon) che necessita di positivi e negativi nello stesso batch.
    """
    def __init__(self, labels: List[int], batch_size: int):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        
        # Indici per classe
        self.cls0_indices = np.where(self.labels == 0)[0]
        self.cls1_indices = np.where(self.labels == 1)[0]
        
        self.n_cls0 = len(self.cls0_indices)
        self.n_cls1 = len(self.cls1_indices)
        
        self.n_batches = int(len(self.labels) / self.batch_size)
        
        self.n_samples_per_class = self.batch_size // 2

    def __iter__(self) -> Iterator[List[int]]:
        np.random.shuffle(self.cls0_indices)
        np.random.shuffle(self.cls1_indices)
        
        ptr0 = 0
        ptr1 = 0
        
        for _ in range(self.n_batches):
            batch_indices = []
            
            end0 = ptr0 + self.n_samples_per_class
            if end0 > self.n_cls0:
                np.random.shuffle(self.cls0_indices)
                ptr0 = 0
                end0 = self.n_samples_per_class
            batch_indices.extend(self.cls0_indices[ptr0:end0])
            ptr0 = end0
            
            end1 = ptr1 + (self.batch_size - len(batch_indices))
            if end1 > self.n_cls1:
                np.random.shuffle(self.cls1_indices)
                ptr1 = 0
                end1 = self.batch_size - len(batch_indices)
            batch_indices.extend(self.cls1_indices[ptr1:end1])
            ptr1 = end1
            
            np.random.shuffle(batch_indices)
            yield batch_indices.tolist()

    def __len__(self) -> int:
        return self.n_batches

# -----------------------------------------------------------------------------
# 2. Dataset Class
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
    """
    Forensic Code Dataset.
    Strategies:
    1. Tabula Rasa (De-Styling): Removes comments & whitespace.
    2. Canonicalization: Abstracts literals ("STR", "NUM") and enforces operator spacing.
    3. Robust Augmentation: Random crops & token masking.
    """
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
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size 
        
        self.regex_comments_c = re.compile(r'//.*|/\*.*?\*/', re.DOTALL)
        self.regex_comments_py = re.compile(r'#.*|""".*?"""', re.DOTALL)
        self.regex_strings = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
        self.regex_numbers = re.compile(r'\b\d+\.?\d*\b|\b0x[0-9a-fA-F]+\b')
        self.regex_operators = re.compile(r'([\[\]\{\}\(\)\,\;\:\+\-\*\/\=\<\>\|\&\%\!])')

    def __len__(self) -> int:
        return len(self.data)

    def _normalize_forensic(self, code: str) -> str:
        """
        Trasforma il codice in una sequenza di token strutturali astratti.
        Rimuove qualsiasi bias di formattazione o contenuto letterale.
        """
        if not isinstance(code, str): return ""

        code = self.regex_comments_c.sub('', code)
        code = self.regex_comments_py.sub('', code)
        
        code = self.regex_strings.sub('"STR"', code)
        code = self.regex_numbers.sub('NUM', code)

        code = self.regex_operators.sub(r' \1 ', code)

        code = re.sub(r'\s+', ' ', code).strip()
        
        return code

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        code_raw = str(self.data.at[idx, "code"])
        label = int(self.data.at[idx, "label"]) 
        lang_str = str(self.data.at[idx, "language"]).lower()
        
        lang_id = self.language_map.get(lang_str, -1)
        
        code = self._normalize_forensic(code_raw)

        if len(code) < 5:
            code = "return void ;"

        if self.augment and len(code) > self.max_length * 4:
            window_size_chars = int(self.max_length * 3.5)
            max_start = len(code) - window_size_chars
            
            if max_start > 0 and random.random() < 0.7:
                start_idx = random.randint(0, max_start)
                while start_idx < len(code) and code[start_idx] != ' ':
                    start_idx += 1
                code = code[start_idx : start_idx + window_size_chars]

        encoding = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        if self.augment:
            probability_matrix = torch.full(input_ids.shape, 0.15)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                input_ids.tolist(), already_has_special_tokens=True
            )
            special_mask_tensor = torch.tensor(special_tokens_mask, dtype=torch.bool)
            probability_matrix.masked_fill_(special_mask_tensor, value=0.0)
            probability_matrix.masked_fill_(attention_mask == 0, value=0.0)
            
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
            input_ids[indices_replaced] = self.mask_token_id
            
            indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(0, self.vocab_size, input_ids.shape, dtype=torch.long)
            input_ids[indices_random] = random_words[indices_random]
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long)
        }

# -----------------------------------------------------------------------------
# 3. Data Loading Utils
# -----------------------------------------------------------------------------
def load_and_preprocess(file_path: str, max_code_chars: int = 20000) -> pd.DataFrame:
    columns = ['code', 'label', 'language']
    try:
        df = pd.read_parquet(file_path, columns=columns)
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise e
    
    df['language'] = df['language'].str.lower()
    df = df.dropna(subset=['code', 'label']).reset_index(drop=True)
    df = df[df['code'].str.strip().str.len() > 15].copy()
    df['code'] = df['code'].str.slice(0, max_code_chars)
    return df

def balance_languages(df: pd.DataFrame, target_samples: int = 3000) -> pd.DataFrame:
    logger.info(f"Balancing languages... Target cap: {target_samples}")
    df_list = []
    if df.empty or 'language' not in df.columns:
        return df

    for lang, count in df['language'].value_counts().items():
        group = df[df['language'] == lang]
        if count > target_samples:
            df_list.append(group.sample(n=target_samples, random_state=42))
        else:
            df_list.append(group)
            
    if not df_list:
        return df
    return pd.concat(df_list).sample(frac=1, random_state=42).reset_index(drop=True)

def load_data(config: dict, tokenizer):
    logger.info(">>> Loading Data for Task A <<<")
    train_path = config["data"]["train_path"]
    val_path = config["data"]["val_path"]
    
    train_df = load_and_preprocess(train_path)
    val_df   = load_and_preprocess(val_path)
    
    samples_cap = config["data"].get("samples_per_lang", 4000)
    if config["data"].get("balance_languages", True):
        train_df = balance_languages(train_df, target_samples=samples_cap)
    
    target_langs = config["model"].get("languages", ["python", "java", "cpp"])
    language_map = {lang.lower(): i for i, lang in enumerate(target_langs)}
    logger.info(f"Language Map Size: {len(language_map)}")
    
    max_len = config["data"].get("max_length", 512)
    
    train_dataset = CodeDataset(train_df, tokenizer, language_map, max_length=max_len, augment=True)
    val_dataset = CodeDataset(val_df, tokenizer, language_map, max_length=max_len, augment=False)
    
    batch_size = config["training"]["batch_size"]
    train_sampler = BalancedBatchSampler(train_dataset.labels_list, batch_size=batch_size)

    logger.info(f"Datasets Ready. Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    return train_dataset, val_dataset, train_sampler, train_df