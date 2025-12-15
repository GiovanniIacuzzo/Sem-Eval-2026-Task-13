import random
import logging
import gc
import pandas as pd
import torch
from typing import Tuple, Dict
from torch.utils.data import Dataset

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
# Label Mapping (Task B)
# -----------------------------------------------------------------------------
# 31 Classes - Critical for Task B
GENERATOR_MAP = {
    '01-ai/yi-coder-1.5b': 0,
    '01-ai/yi-coder-1.5b-chat': 1,
    'bigcode/starcoder': 2,
    'bigcode/starcoder2-15b': 3,
    'bigcode/starcoder2-3b': 4,
    'deepseek-ai/deepseek-coder-1.3b-base': 5,
    'deepseek-ai/deepseek-r1': 6,
    'deepseek-ai/deepseek-v3-0324': 7,
    'gemma-3-27b-it': 8,
    'gemma-3n-e4b-it': 9,
    'google/codegemma-2b': 10,
    'gpt-4o': 11,
    'human': 12,
    'ibm-granite/granite-3.2-2b-instruct': 13,
    'ibm-granite/granite-3.3-8b-base': 14,
    'ibm-granite/granite-3.3-8b-instruct': 15,
    'meta-llama/llama-3.1-8b': 16,
    'meta-llama/llama-3.1-8b-instruct': 17,
    'meta-llama/llama-3.2-11b-vision-instruct': 18,
    'meta-llama/llama-3.2-1b': 19,
    'meta-llama/llama-3.2-3b': 20,
    'microsoft/phi-3-medium-4k-instruct': 21,
    'microsoft/phi-3-mini-4k-instruct': 22,
    'microsoft/phi-3-small-8k-instruct': 23,
    'mistralai/devstral-small-2505': 24,
    'mistralai/mistral-7b-instruct-v0.3': 25,
    'qwen/qwen2.5-72b-instruct': 26,
    'qwen/qwen2.5-codder-14b-instruct': 27,
    'qwen/qwen2.5-coder-1.5b': 28,
    'qwen/qwen2.5-coder-1.5b-instruct': 29,
    'qwen/qwq-32b': 30,
}

# -----------------------------------------------------------------------------
# PyTorch Dataset Class with Sliding Window
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
    """
    Advanced Dataset for Code Classification on Limited Hardware.
    
    Features:
    1. Sliding Window Chunking (Training): Splits long files into multiple samples.
       This dramatically increases effective training data and coverage without OOM.
    2. Head+Tail Truncation (Validation): Keeps the start and end of code for robustness.
    3. Memory Efficient: Does not hold tokenized tensors in RAM; tokenizes on the fly.
    """
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        tokenizer, 
        max_length: int = 512, 
        mode: str = "train", 
        overlap: int = 128
    ):
        """
        Args:
            dataframe: The source data.
            tokenizer: HF Tokenizer.
            max_length: Model context size (use 512 for UniXCoder).
            mode: 'train' (uses sliding window) or 'val' (deterministic).
            overlap: Overlap between chunks in sliding window.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.overlap = overlap
        self.mask_token_id = tokenizer.mask_token_id
        
        # --- Pre-Compute Indices for Sliding Window ---
        # Instead of just storing the dataframe, we store a list of (row_idx, chunk_idx)
        # to map __getitem__ index to a specific chunk of a specific file.
        self.samples = []
        self.data = dataframe.reset_index(drop=True)
        
        logger.info(f"Preparing dataset in {mode} mode...")
        
        # Estimate token count roughly (chars / 3.5) to speed up setup
        # If code is short, we just keep it as 1 chunk.
        # If long, and mode is TRAIN, we split it.
        
        if self.mode == "train":
            for idx, row in self.data.iterrows():
                code_len = len(row['code'])
                # Rough token estimate
                est_tokens = code_len / 3.5 
                
                if est_tokens <= max_length:
                    self.samples.append((idx, 0, -1)) # (row_idx, start_char, end_char) -1 means full
                else:
                    # Create chunks
                    # We work in characters here for speed, assuming avg token is 3-4 chars
                    # Stride in chars
                    chunk_char_len = int(max_length * 3.5)
                    stride = int((max_length - overlap) * 3.5)
                    
                    for start in range(0, code_len, stride):
                        # Don't add tiny chunks at the end
                        if start + 50 < code_len: 
                            self.samples.append((idx, start, start + chunk_char_len))
        else:
            # Validation/Test: No chunking expansion to keep metrics comparable.
            # We will handle long files by Head+Tail truncation inside __getitem__
            for idx in range(len(self.data)):
                self.samples.append((idx, 0, -1))

        logger.info(f"Dataset expansion: {len(self.data)} files -> {len(self.samples)} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row_idx, start, end = self.samples[idx]
        row = self.data.iloc[row_idx]
        
        code = str(row["code"])
        label = int(row["label"])
        
        # --- 1. Slicing Strategy ---
        if self.mode == "train":
            if end != -1:
                # Sliding Window Slice
                text_chunk = code[start:end]
            else:
                text_chunk = code
        else:
            # Validation: Head + Tail Strategy
            # If code is too long, we take the first (MAX/2) and last (MAX/2) characters
            # This captures imports (Head) and return logic (Tail).
            limit = int(self.max_length * 3.5)
            if len(code) > limit:
                half = limit // 2
                text_chunk = code[:half] + "\n...[SNIP]...\n" + code[-half:]
            else:
                text_chunk = code

        # --- 2. Tokenization ---
        # Note: 'truncation=True' handles any remaining excess length
        encoding = self.tokenizer(
            text_chunk,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # --- 3. Augmentation (Train only) ---
        # Low probability MLM to make model robust to variable naming changes
        if self.mode == "train" and random.random() < 0.1: # 10% chance to apply masking
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
# Data Preprocessing Logic
# -----------------------------------------------------------------------------
def load_and_preprocess(
    file_path: str, 
    max_code_len: int = 20000, # Hard char limit to save RAM initially
    task_type: str = "multiclass"
) -> pd.DataFrame:
    """
    Loads Parquet file and applies task-specific preprocessing.
    """
    logger.info(f"Loading data from {file_path} (Task: {task_type})")
    
    columns = ['code', 'language', 'generator']
    if task_type == "binary":
        columns.append('label')
        
    try:
        df = pd.read_parquet(file_path, columns=columns)
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise e
    
    # 1. Clean Data
    df = df.dropna(subset=['code']).reset_index(drop=True)
    df['language'] = df['language'].str.lower()
    
    # 2. Filter useless short snippets (less than 20 chars usually garbage)
    df = df[df['code'].str.len() > 20].copy()
    
    # 3. Hard Cap on Length (Memory Safety)
    # Truncate extremely long files at character level before anything else
    df['code'] = df['code'].str.slice(0, max_code_len)
    
    # 4. Label Mapping
    if task_type == "multiclass":
        if 'generator' not in df.columns:
            raise ValueError("Task B requires 'generator' column.")
            
        df['generator'] = df['generator'].str.lower()
        
        # Filter unknown classes strictly
        known_gens = set(GENERATOR_MAP.keys())
        df = df[df['generator'].isin(known_gens)].copy()
        
        df['label'] = df['generator'].map(GENERATOR_MAP).astype(int)
        
    elif task_type == "binary":
        if 'label' not in df.columns:
             raise ValueError("Task A requires 'label' column (0/1).")
        df['label'] = df['label'].astype(int)
    
    # Explicit garbage collection
    gc.collect()
    
    return df

def balance_languages(df: pd.DataFrame, fraction: float = 1.0, max_samples: int = 2000) -> pd.DataFrame:
    """
    Balances the dataset by capping the maximum samples per language/label pair.
    This prevents Python/Java from dominating rare languages.
    """
    logger.info("Balancing dataset distribution...")
    # Group by both language and label to ensure we don't lose specific generators in specific languages
    df_list = []
    
    # Create stratification key
    df['strat_key'] = df['language'] + "_" + df['label'].astype(str)
    
    for _, group in df.groupby('strat_key'):
        if len(group) > max_samples:
            df_list.append(group.sample(n=max_samples, random_state=42))
        else:
            df_list.append(group)
            
    balanced_df = pd.concat(df_list).sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_df = balanced_df.drop(columns=['strat_key'])
    
    gc.collect()
    return balanced_df

# -----------------------------------------------------------------------------
# Main Data Loading Interface
# -----------------------------------------------------------------------------
def load_data(config: dict, tokenizer) -> Tuple[CodeDataset, CodeDataset, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates data loading pipeline.
    """
    logger.info("Starting Data Pipeline...")
    
    task_type = config["data"].get("task_type", "multiclass")
    max_len  = config["data"].get("max_length", 512) # Increased to 512 for better context

    # 1. Load Data
    train_df = load_and_preprocess(config["data"]["train_path"], task_type=task_type)
    val_df   = load_and_preprocess(config["data"]["val_path"], task_type=task_type)
    
    # 2. Balancing (Optional but Recommended)
    if config["data"].get("balance_languages", False):
        train_df = balance_languages(train_df, max_samples=config["data"].get("samples_per_group", 3000))

    # 3. Demo/Debug Mode
    if config.get("demo", {}).get("active", False):
        logger.warning("DEMO MODE: Using 10% of data")
        train_df = train_df.sample(frac=0.1).reset_index(drop=True)
        val_df = val_df.sample(frac=0.1).reset_index(drop=True)

    # 4. Create Datasets with Different Modes
    # Train gets Sliding Window (Data Augmentation)
    train_dataset = CodeDataset(
        train_df, 
        tokenizer, 
        max_length=max_len, 
        mode="train",
        overlap=128
    )
    
    # Val gets Head+Tail (Deterministic Evaluation)
    val_dataset = CodeDataset(
        val_df, 
        tokenizer, 
        max_length=max_len, 
        mode="val"
    )

    logger.info(f"Final Setup -> Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset, train_df, val_df