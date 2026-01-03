import os
import random
import logging
import pandas as pd
import torch
import numpy as np
import re
from typing import Dict, Tuple, List
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------------------------------------------------------
# Logger Setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Pre-computation Utilities (Performance Optimization)
# -----------------------------------------------------------------------------
def precompute_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes stylometric features once for the entire dataframe.
    Much faster than doing it inside __getitem__.
    """
    logger.info("Pre-computing stylistic features...")

    # We use a vectorized approach where possible or apply simply
    # Because this runs once at startup, efficiency is less critical than inside the loop,
    # but apply is still faster than doing it per-item during training.
    
    def _extract_single(code):
        if not isinstance(code, str): code = str(code)
        
        lines = code.split('\n')
        num_lines = len(lines) if len(lines) > 0 else 1
        code_len = len(code) + 1
        words = len(code.split()) + 1
        
        # 1. Comment Density (heuristic)
        comments = len(re.findall(r'(//|#|/\*|--)', code))
        comment_density = comments / words
        
        # 2. Space Ratio (Whitespace usage)
        space_ratio = code.count(' ') / code_len
        
        # 3. Avg Line Length
        avg_line_len = (code_len / num_lines) / 100.0  # Normalized roughly
        
        # 4. Special Character Density
        special_chars_count = len(re.findall(r'[!@#$%^&*()\-+={}\[\]|\\:;"\'<>,.?/]', code))
        special_chars = special_chars_count / code_len

        return [comment_density, space_ratio, avg_line_len, special_chars]

    # Create a new column 'style_feats_vec' containing the list of 4 floats
    df['style_feats_vec'] = df['code'].apply(_extract_single)
    
    logger.info("Stylistic features computed.")
    return df

# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
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

    def _structural_noise(self, code: str) -> str:
        """Applies random structural noise for data augmentation."""
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            # 5% chance to insert an empty line
            if random.random() < 0.05: new_lines.append("")
            # 5% chance to append trailing spaces
            if random.random() < 0.05: line = line + " " * random.randint(1, 3)
            new_lines.append(line)
        return "\n".join(new_lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        code = str(row["code"])
        label = int(row["label"]) 
        lang_str = str(row["language"]).lower()
        lang_id = self.language_map.get(lang_str, -1)
        
        # 1. Augmentation (Training Only)
        if self.augment:
            code = self._structural_noise(code)

        # 2. Tokenization with Head + Tail Strategy
        # We handle special tokens manually for precision with truncation
        full_tokens = self.tokenizer.encode(code, add_special_tokens=True) 
        
        if len(full_tokens) > self.max_length:
            # Head + Tail Truncation
            # We keep CLS (index 0) and SEP (index -1) implicitly by how we slice
            # Logic: [CLS] + Head + Tail + [SEP]
            
            # Allow space for CLS and SEP
            remaining_capacity = self.max_length - 2 
            half = remaining_capacity // 2
            
            # Head: Skip CLS (0), take 'half' tokens
            head_tokens = full_tokens[1 : 1 + half]
            
            # Tail: Take last 'half' tokens, excluding SEP (-1)
            tail_tokens = full_tokens[-(half + 1) : -1]
            
            input_ids = [self.tokenizer.cls_token_id] + head_tokens + tail_tokens + [self.tokenizer.sep_token_id]
        else:
            # Standard padding handled below
            input_ids = full_tokens

        # 3. Manual Padding (Robust)
        # Ensure we don't exceed max_length (safety check)
        input_ids = input_ids[:self.max_length]
        
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            
        # Convert to Tensors
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids_tensor != self.tokenizer.pad_token_id).long()
        
        # 4. Stylometric Features (Pre-computed)
        # Retrieve from dataframe column (fast)
        style_feats = torch.tensor(row['style_feats_vec'], dtype=torch.float32)
        
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask,
            "style_feats": style_feats,
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long)
        }

    def __len__(self) -> int:
        return len(self.data)

# -----------------------------------------------------------------------------
# Data Loading & Balancing Utils
# -----------------------------------------------------------------------------
def load_and_preprocess(file_path: str, max_code_chars: int = 15000) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    logger.info(f"Reading parquet file: {file_path}")
    df = pd.read_parquet(file_path, columns=['code', 'label', 'language'])
    
    # Basic Cleaning
    df['language'] = df['language'].str.lower()
    df = df.dropna(subset=['code']).reset_index(drop=True)
    
    # Filter extremely short snippets (often noise)
    df = df[df['code'].str.len() > 10].copy()
    
    # Hard clip char length to save memory before tokenization
    df['code'] = df['code'].str.slice(0, max_code_chars)
    
    return df

def get_smart_subset(df: pd.DataFrame, max_total: int = 200000) -> pd.DataFrame:
    """
    Advanced Subsetting:
    1. Keeps ALL rare classes (Hybrid/Adversarial).
    2. Subsamples common classes (Human/Machine).
    3. Tries to balance LANGUAGES within Human/Machine to avoid Python bias.
    """
    logger.info(f"Applying Smart Subsetting/Balancing (Target: ~{max_total})...")
    
    # Separate classes
    df_hybrid = df[df['label'] == 2]
    df_adversarial = df[df['label'] == 3]
    df_human = df[df['label'] == 0]
    df_machine = df[df['label'] == 1]
    
    # Always keep all rare examples
    special_count = len(df_hybrid) + len(df_adversarial)
    remaining_budget = max_total - special_count
    
    if remaining_budget <= 0:
        logger.warning("Target size too small for rare classes. Returning rare only.")
        return pd.concat([df_hybrid, df_adversarial]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split remaining budget between Human and Machine
    target_per_class = remaining_budget // 2
    
    def _stratified_lang_sample(sub_df, n_target):
        if len(sub_df) <= n_target:
            return sub_df
        
        # Calculate weights to penalize over-represented languages (like Python/Java)
        lang_counts = sub_df['language'].value_counts()
        weights = 1.0 / lang_counts
        
        # Assign weight to each row
        sample_weights = sub_df['language'].map(weights)
        
        # Sample with weights
        return sub_df.sample(n=n_target, weights=sample_weights, random_state=42)

    # Sample Human and Machine balancing languages
    df_human_bal = _stratified_lang_sample(df_human, target_per_class)
    df_machine_bal = _stratified_lang_sample(df_machine, target_per_class)
    
    logger.info(f"Subset Stats -> Human: {len(df_human_bal)}, Machine: {len(df_machine_bal)}, "
                f"Hybrid: {len(df_hybrid)}, Adv: {len(df_adversarial)}")

    final_df = pd.concat([df_hybrid, df_adversarial, df_human_bal, df_machine_bal])
    return final_df.sample(frac=1, random_state=42).reset_index(drop=True)

def get_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    labels = df['label'].values
    classes = np.unique(labels)
    # Using 'balanced' handles the remaining imbalance after subsetting
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    
    # Safety check for missing classes
    if len(weights) < 4:
        logger.warning(f"Warning: Only {len(weights)} classes found for weighting. Padding with 1.0.")
        full_weights = np.ones(4)
        for i, c in enumerate(classes):
            if c < 4: full_weights[c] = weights[i]
        weights = full_weights
        
    return torch.tensor(weights, dtype=torch.float32).to(device)

def get_dynamic_language_map(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, int]:
    """Combines languages from train and val to ensure consistent mapping."""
    langs_train = set(train_df['language'].unique())
    langs_val = set(val_df['language'].unique())
    unique_langs = sorted(list(langs_train.union(langs_val)))
    
    mapping = {lang: i for i, lang in enumerate(unique_langs)}
    logger.info(f"Language Map Created: {len(mapping)} languages.")
    return mapping

def load_data_for_training(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Main entry point for data loading.
    Returns: (train_df, val_df, language_map)
    """
    logger.info(">>> Loading Data <<<")
    
    # 1. Load separately
    train_df = load_and_preprocess(config["data"]["train_path"])
    val_df = load_and_preprocess(config["data"]["val_path"])
    
    # 2. Apply Smart Subsetting to TRAIN only (Validation must remain pure)
    max_train_samples = config["data"].get("max_training_samples", 200000)
    train_df = get_smart_subset(train_df, max_total=max_train_samples)
    
    # 3. Precompute Features
    train_df = precompute_style_features(train_df)
    val_df = precompute_style_features(val_df)
    
    # 4. Create consistent language map
    language_map = get_dynamic_language_map(train_df, val_df)
    
    logger.info(f"Final sizes -> Train: {len(train_df)} | Val: {len(val_df)}")
    
    return train_df, val_df, language_map