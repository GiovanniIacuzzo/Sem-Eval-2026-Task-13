import torch
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class AgnosticDataset(Dataset):
    """
    Dataset ottimizzato per SemEval Task A.
    Gestisce input ibridi: Token IDs (Semantici) + Features Manuali (Stilometriche).
    Include Data Augmentation (Random Cropping) e Feature Normalization.
    """
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 512, is_train: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        self.df = dataframe.reset_index(drop=True)
        
        required_cols = {'code', 'label', 'agnostic_features'}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"Dataframe missing required columns: {required_cols - set(self.df.columns)}")

        features_list = self.df['agnostic_features'].tolist()
        self.features_matrix = np.array(features_list, dtype=np.float32)
        
        self.num_samples = len(self.df)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        logger.info(f"Dataset Initialized | Split: {'TRAIN' if is_train else 'VAL/TEST'} | Samples: {self.num_samples}")

    def __len__(self):
        return self.num_samples

    def _normalize_features(self, feature_vector: torch.Tensor) -> torch.Tensor:
        """
        Applica trasformazioni specifiche per portare le feature in range compatibili.
        Mapping basato su preprocess_features.py:
        idx 0: Perplexity (Unbounded -> Log)
        idx 1: Avg ID Len (Unbounded -> Log)
        idx 7: Line Len Std (Unbounded -> Log)
        Altri: Ratio 0-1 (Keep as is)
        """
        x = feature_vector.clone()
        
        indices_to_log = [0, 1, 7] 
        
        for i in indices_to_log:
            if i < x.shape[0]:
                x[i] = torch.log1p(x[i])
        
        x = torch.clamp(x, min=0.0, max=100.0)
        return x

    def __getitem__(self, idx):
        code = str(self.df.iat[idx, self.df.columns.get_loc('code')])
        label = int(self.df.iat[idx, self.df.columns.get_loc('label')])
        
        raw_feats = self.features_matrix[idx]
        feats_tensor = torch.tensor(raw_feats, dtype=torch.float32)
        norm_feats = self._normalize_features(feats_tensor)

        input_ids = self.tokenizer.encode(code, add_special_tokens=True, truncation=False)
        total_len = len(input_ids)
        
        if total_len > self.max_length:
            if self.is_train:
                start_token = input_ids[0]
                
                max_start_idx = total_len - self.max_length + 1
                random_start = np.random.randint(1, max_start_idx)
                
                cropped_ids = [start_token] + input_ids[random_start : random_start + self.max_length - 1]
                final_input_ids = cropped_ids
            else:
                final_input_ids = input_ids[:self.max_length]
        else:
            final_input_ids = input_ids
            
        processed_len = len(final_input_ids)
        padding_needed = self.max_length - processed_len
        
        if padding_needed > 0:
            final_input_ids = final_input_ids + [self.pad_token_id] * padding_needed
            attention_mask = [1] * processed_len + [0] * padding_needed
        else:
            attention_mask = [1] * self.max_length

        return {
            "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "extra_features": norm_feats,
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_data(config, tokenizer):
    """Factory function per caricare train e val dataset."""
    data_dir = config["data"]["data_dir"]
    
    logger.info("Loading Parquet files...")
    train_df = pd.read_parquet(f"{data_dir}/train_processed.parquet")
    val_df = pd.read_parquet(f"{data_dir}/val_processed.parquet")
    
    train_df = train_df.dropna(subset=['label']).reset_index(drop=True)
    val_df = val_df.dropna(subset=['label']).reset_index(drop=True)
    
    max_len = config["data"]["max_length"]
    
    train_ds = AgnosticDataset(train_df, tokenizer, max_length=max_len, is_train=True)
    val_ds = AgnosticDataset(val_df, tokenizer, max_length=max_len, is_train=False)
    
    return train_ds, val_ds