import os
import random
import logging
import pandas as pd
import torch
import seaborn as sns
import seaborn as plt
from typing import Tuple, List, Dict, Optional
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
# PyTorch Dataset Class
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
    """
    Custom PyTorch Dataset for Source Code Classification.
    
    Updated for DANN (Domain Adversarial Training):
    - Returns 'lang_ids' alongside input_ids and labels.
    - Maps language strings to integers based on a provided mapping.
    """
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 tokenizer, 
                 language_map: Dict[str, int],
                 max_length: int = 256, 
                 augment: bool = False):
        
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.language_map = language_map  # {'python': 0, 'java': 1, ...}
        self.max_length = max_length
        self.augment = augment
        
        # Cache mask token ID for performance during augmentation
        self.mask_token_id = tokenizer.mask_token_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        code = str(self.data.loc[idx, "code"])
        label = int(self.data.loc[idx, "label"])
        lang_str = str(self.data.loc[idx, "language"]).lower()
        
        # DANN Target: Map string to ID. Return -1 if language is unknown (unseen during training)
        lang_id = self.language_map.get(lang_str, -1)
        
        # ---------------------------------------------------------
        # 1. Random Cropping Strategy (Generalization)
        # ---------------------------------------------------------
        if self.augment and len(code) > self.max_length * 4:
            start_idx = random.randint(0, len(code) - self.max_length * 4)
            code = code[start_idx:] 

        # ---------------------------------------------------------
        # 2. Tokenization
        # ---------------------------------------------------------
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # ---------------------------------------------------------
        # 3. Smart Augmentation (Token-Level Masking)
        # ---------------------------------------------------------
        if self.augment:
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
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long) # New for DANN
        }

# -----------------------------------------------------------------------------
# Data Preprocessing Logic
# -----------------------------------------------------------------------------
def load_and_preprocess(file_path: str, max_code_len: int = 4096) -> pd.DataFrame:
    """Loads Parquet file and applies initial cleaning."""
    columns = ['code', 'label', 'language', 'generator']
    try:
        df = pd.read_parquet(file_path, columns=columns)
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise e
    
    df['language'] = df['language'].str.lower()
    df = df.dropna(subset=['code'])
    df = df[df['code'].str.len() > 10].copy()
    df['code'] = df['code'].str.slice(0, max_code_len)
    
    return df

def stratified_sample(df: pd.DataFrame, fraction: float = 0.3, stratify_cols: List[str] = ["language", "label"], random_state: int = 42) -> pd.DataFrame:
    stratify_key = df[stratify_cols].astype(str).agg("__".join, axis=1)
    sampled_df, _ = train_test_split(
        df, train_size=fraction, stratify=stratify_key, random_state=random_state
    )
    return sampled_df

def balance_languages(df: pd.DataFrame, fraction: float = 0.3, max_samples_per_lang: int = 5000, random_state: int = 42) -> pd.DataFrame:
    df_list = []
    for _, group in df.groupby('language'):
        n_samples = int(len(group) * fraction)
        n_samples = min(n_samples, max_samples_per_lang)
        if n_samples > 0:
            df_list.append(group.sample(n=n_samples, random_state=random_state))
            
    if not df_list:
        return df
        
    return pd.concat(df_list).sample(frac=1, random_state=random_state).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------------------
def eda_dataframe(df: pd.DataFrame, name: str, img_path: str):
    os.makedirs(img_path, exist_ok=True)
    lengths = df['code'].str.len()
    
    plt.figure(figsize=(10,4))
    sns.histplot(x=lengths, hue=df['label'], bins=50, log_scale=True)
    plt.savefig(os.path.join(img_path, f"{name}_length.png"))
    plt.close()

    plt.figure(figsize=(10,4))
    sns.countplot(data=df, x='language', hue='label')
    plt.savefig(os.path.join(img_path, f"{name}_lang.png"))
    plt.close()

# -----------------------------------------------------------------------------
# Main Data Loading Interface
# -----------------------------------------------------------------------------
def load_data(config: dict, tokenizer) -> Tuple[CodeDataset, CodeDataset, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates data loading, preprocessing, balancing, and dataset creation.
    """
    logger.info("Starting Data Pipeline...")
    
    train_df = load_and_preprocess(config["data"]["train_path"])
    val_df   = load_and_preprocess(config["data"]["val_path"])
    
    max_len  = config["data"].get("max_length", 256)

    # 1. Generate Language Mapping for DANN
    # We use the list from config to ensure consistent ID mapping
    target_languages = config["model"].get("languages", ["python", "java", "c++"])
    # Map: {'python': 0, 'java': 1, 'c++': 2}
    language_map = {lang.lower(): i for i, lang in enumerate(target_languages)}
    logger.info(f"DANN Language Mapping: {language_map}")

    # 2. Class Balancing
    if config["data"].get("balance_languages", True):
        logger.info("Applying Language Balancing...")
        train_df = balance_languages(train_df)
        val_df   = balance_languages(val_df) 
    
    # 3. Demo Mode
    demo_cfg = config.get("demo", {})
    if demo_cfg.get("active", False):
        fraction = demo_cfg.get("fraction", 0.3)
        train_df = stratified_sample(train_df, fraction=fraction)
        val_df   = stratified_sample(val_df, fraction=fraction)

    logger.info(f"Final Dataset Sizes -> Train: {len(train_df)} | Val: {len(val_df)}")
    
    # 4. Dataset Instantiation with Language Map
    train_dataset = CodeDataset(train_df, tokenizer, language_map, max_length=max_len, augment=True)
    val_dataset   = CodeDataset(val_df, tokenizer, language_map, max_length=max_len, augment=False)

    return train_dataset, val_dataset, train_df, val_df

# -----------------------------------------------------------------------------
# Unit Testing
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    print("\n[Self-Test] verifying Dataset Logic...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        data = {
            'code': ["print('hello')", "int x = 0;"],
            'label': [0, 1],
            'language': ['python', 'c++'], # 'c++' exists in map below
            'generator': ['human', 'gpt']
        }
        df = pd.DataFrame(data)
        
        # Define mock language map
        lang_map = {'python': 0, 'c++': 1}
        
        ds = CodeDataset(df, tokenizer, lang_map, max_length=32, augment=True)
        item = ds[0]
        
        print(f"Input IDs Shape: {item['input_ids'].shape}")
        print(f"Lang ID: {item['lang_ids']} (Expected: 0 for python)")
        
        # Test unknown language
        data_unknown = {'code': ["echo 'hi'"], 'label': [0], 'language': ['bash'], 'generator': ['human']}
        ds_unk = CodeDataset(pd.DataFrame(data_unknown), tokenizer, lang_map, max_length=32)
        print(f"Unknown Lang ID: {ds_unk[0]['lang_ids']} (Expected: -1)")
        
        print("Dataset Logic Valid.")
        
    except Exception as e:
        print(f"Dataset Logic Failed: {e}")