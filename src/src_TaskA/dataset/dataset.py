import torch
import os
import logging
import math
import re
import numpy as np
import collections
from typing import List, Dict, Optional
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# =============================================================================
# 1. STYLOMETRIC ENGINE
# =============================================================================
class StylometricEngine:
    """
    Language-Agnostic Feature Extraction Engine.
    Computes entropy, consistency, and structural metrics to distinguish 
    Human organic code from LLM synthetic code.
    Used by extract_features.py and inference.py.
    """
    def __init__(self):
        self.keywords = {
            "def", "class", "return", "if", "else", "elif", "for", "while", "import", "from",
            "try", "except", "finally", "with", "as", "pass", "break", "continue", "lambda",
            "global", "nonlocal", "assert", "del", "yield", "raise", "print", "true", "false",
            "none", "int", "float", "str", "bool", "list", "dict", "set", "tuple", "void", 
            "public", "private", "protected", "static", "final", "const", "var", "let", 
            "function", "package", "namespace", "using", "struct", "impl", "interface"
        }

    def process(self, code: str, perplexity: float) -> List[float]:
        identifiers = self._extract_identifiers(code)
        
        # 1. Entropy & Naming
        entropy = self._calculate_entropy(identifiers)
        avg_len = sum(len(x) for x in identifiers) / len(identifiers) if identifiers else 0.0
        
        # 2. Consistency (Snake vs Camel)
        snake_case = sum(1 for x in identifiers if '_' in x)
        camel_case = sum(1 for x in identifiers if '_' not in x and any(c.isupper() for c in x))
        total_style = snake_case + camel_case
        mix_ratio = (min(snake_case, camel_case) / total_style) if total_style > 0 else 0.0
        
        # 3. Structural Irregularities
        lines = code.split('\n')
        empty_lines_var = self._get_empty_line_variance(lines)
        
        # 4. Dirty Code Indicators
        code_lower = code.lower()
        dirty_markers = ["todo", "fix", "hack", "tmp", "temp", "debug", "???", "!!!"]
        dirty_score = sum(code_lower.count(m) for m in dirty_markers)
        
        # 5. Repetition (Type-Token Ratio)
        tokens = re.findall(r'\b\w+\b', code_lower)
        ttr = (len(set(tokens)) / len(tokens)) if tokens else 0.0

        # Feature Vector Construction (10 Dimensions)
        return [
            perplexity,               # 0. Perplexity (External)
            math.log1p(avg_len),      # 1. ID Length
            entropy,                  # 2. ID Entropy
            mix_ratio,                # 3. Style Inconsistency
            math.log1p(dirty_score),  # 4. "Dirty" Comments
            empty_lines_var,          # 5. Layout Variance
            ttr,                      # 6. Repetition
            float(len(lines)),        # 7. LOC
            len(tokens) / (len(lines)+1) if len(lines) > 0 else 0, # 8. Density
            math.log1p(total_style)   # 9. Variable Count
        ]

    def _extract_identifiers(self, code: str) -> List[str]:
        candidates = re.findall(r'\b[a-zA-Z_]\w*\b', code)
        return [c for c in candidates if c.lower() not in self.keywords and len(c) > 1]

    def _calculate_entropy(self, identifiers: List[str]) -> float:
        text = "".join(identifiers)
        if not text: return 0.0
        counts = collections.Counter(text)
        total = len(text)
        return -sum((c / total) * math.log2(c / total) for c in counts.values())

    def _get_empty_line_variance(self, lines: List[str]) -> float:
        empty_indices = [i for i, line in enumerate(lines) if not line.strip()]
        if len(empty_indices) < 2: return 0.0
        diffs = np.diff(empty_indices)
        return float(np.var(diffs))

# =============================================================================
# 2. VECTORIZED DATASET
# =============================================================================
class VectorizedDataset(Dataset):
    """
    Dataset wrapper efficiente per tensori pre-calcolati (.pt).
    Usato SOLO da train.py.
    """
    def __init__(self, 
                 data_dict: Dict[str, torch.Tensor], 
                 feature_stats: Optional[Dict[str, torch.Tensor]] = None):
        """
        Args:
            data_dict: Dizionario contenente 'embeddings', 'features', 'labels'.
            feature_stats: Dizionario con 'mean' e 'std' calcolati sul training set.
        """
        self.embeddings = data_dict['embeddings'].float() # [N, 768]
        self.features = data_dict['features'].float()     # [N, 10]
        self.labels = data_dict['labels'].long()          # [N]
        
        # Normalizzazione Z-Score
        if feature_stats is None:
            # Training Mode: Calcola statistiche
            self.mean = self.features.mean(dim=0)
            self.std = self.features.std(dim=0) + 1e-6 
            self.stats = {'mean': self.mean, 'std': self.std}
            logger.info("Calculated Z-Score stats on Training Data.")
        else:
            # Val/Test Mode: Usa statistiche fornite
            self.mean = feature_stats['mean']
            self.std = feature_stats['std']
            self.stats = feature_stats
            # logger.info("Applied external Z-Score stats.")
            
        self.features = (self.features - self.mean) / self.std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "semantic_embedding": self.embeddings[idx],
            "structural_features": self.features[idx],
            "labels": self.labels[idx]
        }

# =============================================================================
# 3. LOADING FUNCTION
# =============================================================================
def load_vectorized_data(config: Dict, holdout_language: Optional[str] = None):
    """
    Carica i file .pt, unisce i dati e crea gli split Train/Val basati sulla logica LOLO.
    """
    data_dir = config.get("vector_data_dir", "data/Task_A/processed")
    
    train_path = os.path.join(data_dir, "train_vectors.pt")
    val_path = os.path.join(data_dir, "val_vectors.pt")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Vector files not found in {data_dir}. Run extract_features.py first.")

    logger.info(f"Loading vectors from {data_dir}...")
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    
    # Merge dei dati
    full_embeddings = torch.cat([train_data['embeddings'], val_data['embeddings']], dim=0)
    full_features = torch.cat([train_data['features'], val_data['features']], dim=0)
    full_labels = torch.cat([train_data['labels'], val_data['labels']], dim=0)
    full_languages = np.concatenate([train_data['languages'], val_data['languages']], axis=0)
    
    total_samples = len(full_labels)
    available_langs = np.unique(full_languages)
    logger.info(f"Total Samples: {total_samples} | Languages Found: {available_langs}")

    # Split Logic
    if holdout_language:        
        langs_lower = np.char.lower(full_languages.astype(str))
        target_lower = holdout_language.lower().strip()
        
        if target_lower not in [l.lower() for l in available_langs]:
             raise ValueError(f"Holdout language '{holdout_language}' not found in dataset.")
        
        logger.info(f"Applying LOLO Split: Holding out '{holdout_language}'")
        val_mask = (langs_lower == target_lower)
        train_mask = ~val_mask
    else:
        logger.warning("No holdout language specified. Using random 90/10 split.")
        indices = np.random.permutation(total_samples)
        split_point = int(total_samples * 0.9)
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]
        
        train_mask = np.zeros(total_samples, dtype=bool)
        val_mask = np.zeros(total_samples, dtype=bool)
        train_mask[train_indices] = True
        val_mask[val_indices] = True

    # Helper filter
    def filter_data(mask):
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        return {
            'embeddings': full_embeddings[mask_tensor],
            'features': full_features[mask_tensor],
            'labels': full_labels[mask_tensor]
        }
    
    train_dict = filter_data(train_mask)
    val_dict = filter_data(val_mask)
    
    logger.info(f"Train Set Size: {len(train_dict['labels'])} | Val Set Size: {len(val_dict['labels'])}")

    # Istanziazione
    train_dataset = VectorizedDataset(train_dict, feature_stats=None)
    val_dataset = VectorizedDataset(val_dict, feature_stats=train_dataset.stats)
    
    return train_dataset, val_dataset