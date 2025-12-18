import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, mode="binary"):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode # 'binary' o 'families' o 'validation'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = str(self.data.at[idx, 'code'])
        
        # Logica Labeling
        if self.mode == "binary":
            # Per il training binario usiamo la colonna 'is_ai' creata nello Step 1
            label = int(self.data.at[idx, 'is_ai'])
        
        elif self.mode == "families":
            # Per il training famiglie usiamo la colonna 'family_label' (0-9)
            label = int(self.data.at[idx, 'family_label'])
            
        else: # Validation / Inference su dati grezzi
            # Qui la logica è tricky. Se stiamo validando il modello Binary:
            # Vogliamo che label sia 0 (Human) o 1 (AI) derivato dalla label originale
            original = int(self.data.at[idx, 'label'])
            if original == -1: label = -1
            else:
                # Se il task è binary validation:
                # 0 -> 0, 1..10 -> 1
                # Se il task è family validation:
                # Questo dataset NON dovrebbe contenere Human. 
                # Se li contiene, li ignoriamo o gestiamo esternamente.
                # Assumiamo per ora validation standard:
                label = original 

        # Tokenization
        encoding = self.tokenizer(
            code, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_data(config, tokenizer, mode="binary"):
    """
    Carica i dati in base alla modalità (binary o families).
    """
    common_cfg = config["common"]
    specific_cfg = config[mode]
    
    data_dir = common_cfg["data_dir"]
    train_file = specific_cfg["train_file"]
    
    # 1. Carica Train
    train_path = os.path.join(data_dir, train_file)
    print(f"Loading Train Data from: {train_path}")
    train_df = pd.read_parquet(train_path)
    
    # 2. Carica Validation (Originale)
    val_path = common_cfg["val_path"]
    print(f"Loading Val Data from: {val_path}")
    val_df = pd.read_parquet(val_path)
    
    # FILTRAGGIO VALIDATION
    # Se siamo in mode 'families', il validation set NON deve avere umani per calcolare le metriche correttamente durante il training
    if mode == "families":
        val_df = val_df[val_df['label'] != 0].copy()
        # Remap labels validation 1-10 -> 0-9
        val_df['family_label'] = val_df['label'] - 1
    elif mode == "binary":
        # Crea target binario sul validation
        val_df['is_ai'] = val_df['label'].apply(lambda x: 0 if x == 0 else 1)

    # 3. Class Weights (Solo per Families)
    class_weights_tensor = None
    if specific_cfg.get("class_weights", False) and mode == "families":
        labels = train_df['family_label'].values
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Dataset Creation
    train_ds = CodeDataset(train_df, tokenizer, common_cfg["max_length"], mode=mode)
    
    # Validation Dataset deve sapere in che modalità siamo per restituire la label giusta
    val_ds = CodeDataset(val_df, tokenizer, common_cfg["max_length"], mode=mode)

    return train_ds, val_ds, class_weights_tensor