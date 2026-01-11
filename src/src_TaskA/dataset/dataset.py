import torch
import os
import logging
import numpy as np
from typing import Dict
from torch.utils.data import Dataset

logger = logging.getLogger("DatasetLoader")

class VectorizedDataset(Dataset):
    """Dataset wrapper ottimizzato per caricare tensori pre-calcolati."""
    
    def __init__(self, data_dict, stats=None):
        self.embeddings = data_dict['embeddings'].float() # [N, 768]
        self.features = data_dict['features'].float()     # [N, 9]
        self.labels = data_dict['labels'].long()          # [N]
        
        # Normalizzazione Z-Score delle features manuali
        if stats is None:
            self.mean = self.features.mean(dim=0)
            self.std = self.features.std(dim=0) + 1e-6 
            self.stats = {'mean': self.mean, 'std': self.std}
        else:
            self.mean = stats['mean']
            self.std = stats['std']
            self.stats = stats
            
        self.features = (self.features - self.mean) / self.std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "semantic_embedding": self.embeddings[idx],
            "structural_features": self.features[idx],
            "labels": self.labels[idx]
        }

def load_vectorized_data(config: Dict):
    """
    Carica i .pt e prepara i dataset Train/Val.
    Accetta il dizionario 'config' invece della stringa raw.
    """
    data_dir = config.get("vector_data_dir", "data/Task_A/processed")
    
    train_path = os.path.join(data_dir, "train_vectors.pt")
    val_path = os.path.join(data_dir, "val_vectors.pt") 
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train vectors not found at {train_path}")
        
    logger.info(f"Loading Training Data: {train_path}")
    train_data = torch.load(train_path)
    
    if os.path.exists(val_path):
        logger.info(f"Loading Validation Data: {val_path}")
        val_data = torch.load(val_path)
    else:
        logger.warning("Validation file not found! Splitting training data 90/10.")
        total = len(train_data['labels'])
        split = int(total * 0.9)
        
        # Split manuale del dizionario
        val_data = {k: v[split:] for k, v in train_data.items() if isinstance(v, (torch.Tensor, np.ndarray))}
        train_data = {k: v[:split] for k, v in train_data.items() if isinstance(v, (torch.Tensor, np.ndarray))}

    # Creazione Dataset
    train_ds = VectorizedDataset(train_data, stats=None)
    val_ds = VectorizedDataset(val_data, stats=train_ds.stats)
    
    logger.info(f"Train Size: {len(train_ds)} | Val Size: {len(val_ds)}")
    return train_ds, val_ds