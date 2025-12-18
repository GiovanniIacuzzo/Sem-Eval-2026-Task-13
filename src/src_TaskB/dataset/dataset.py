import torch
import pandas as pd
import numpy as np
import os
import re
import random
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------------------------------------------------------
# DANN Language Map
# -----------------------------------------------------------------------------
# Mappa fissa per convertire stringhe linguaggio in ID numerici per il DANN.
# L'ID -1 viene usato per "unknown" e ignorato dalla loss.
LANG_MAP = {
    "python": 0, "java": 1, "cpp": 2, "c": 3, "c#": 4, "cs": 4, 
    "javascript": 5, "php": 6, "ruby": 7, "rust": 8, "go": 9,
    "typescript": 10, "kotlin": 11, "swift": 12, "scala": 13, "shell": 14
}

# -----------------------------------------------------------------------------
# Structural Augmentation (Cruciale per classi piccole)
# -----------------------------------------------------------------------------
def augment_code(code: str) -> str:
    """
    Applica perturbazioni strutturali che non cambiano la semantica (molto).
    Utile per Contrastive Learning e per evitare overfitting su dataset piccoli.
    """
    if random.random() > 0.6: # 40% chance di nessuna modifica
        return code

    # 1. Variable Renaming (Naive)
    # Cerca token che sembrano variabili semplici (es. i, count, temp)
    # e li sostituisce con varianti (es. var_X). 
    # Usiamo una regex conservativa per non rompere keywords.
    if random.random() > 0.5:
        # Trova parole di 1-3 lettere minuscole
        targets = list(set(re.findall(r'\b[a-z]{1,3}\b', code)))
        if targets:
            to_rename = random.choice(targets)
            new_name = f"v_{random.randint(10,99)}"
            # Sostituzione word boundary
            code = re.sub(r'\b' + to_rename + r'\b', new_name, code)

    # 2. White Noise (Newlines & Spaces)
    if random.random() > 0.5:
        lines = code.split('\n')
        # Inserisce una riga vuota o un commento a caso
        idx = random.randint(0, len(lines)-1) if lines else 0
        noise = "\n" if random.random() > 0.5 else " # code block\n"
        if lines:
            lines.insert(idx, noise)
            code = "\n".join(lines)
            
    return code

# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, mode="binary", is_train=False):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.is_train = is_train

        # Pre-calcolo Lang IDs per velocità (DANN)
        # Se 'language' non c'è, mettiamo 'unknown'
        if 'language' not in self.data.columns:
            self.data['language'] = 'unknown'
            
        self.data['lang_id'] = self.data['language'].apply(
            lambda x: LANG_MAP.get(str(x).lower().strip(), -1)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = str(self.data.at[idx, 'code'])
        lang_id = int(self.data.at[idx, 'lang_id'])
        
        # LABELING LOGIC
        if self.mode == "binary":
            label = int(self.data.at[idx, 'is_ai'])
        elif self.mode == "families":
            label = int(self.data.at[idx, 'family_label'])
        else:
            # Fallback generico
            label = int(self.data.at[idx, 'label'])

        # AUGMENTATION (Solo Training)
        if self.is_train:
            code = augment_code(code)

        # TOKENIZATION
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
            "labels": torch.tensor(label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long) # CRITICO PER DANN
        }

# -----------------------------------------------------------------------------
# Data Loader Function
# -----------------------------------------------------------------------------
def load_data(config, tokenizer, mode="binary"):
    """
    Carica i dati specifici per la modalità richiesta.
    Usa i file pre-processati (train_X.parquet, val_X.parquet).
    """
    common_cfg = config["data"] # Nota: config["data"] non ["common"] nel tuo yaml finale
    specific_cfg = config["training"] # Parametri training generici
    
    # Costruzione percorsi file basati sulla modalità
    # Assumiamo che prepare_datasets.py abbia creato questi file
    data_dir = common_cfg.get("data_dir", "data/Task_B_Processed") # Default safe
    
    train_file = f"train_{mode}.parquet"
    val_file = f"val_{mode}.parquet"
    
    train_path = os.path.join(data_dir, train_file)
    val_path = os.path.join(data_dir, val_file)
    
    print(f"Loading Train: {train_path}")
    print(f"Loading Val:   {val_path}")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"File dati mancanti in {data_dir}. Esegui prima prepare_datasets.py!")

    # Caricamento DataFrame
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # Calcolo Class Weights (Solo per Families Training)
    class_weights_tensor = None
    use_weights = config["model"].get("class_weights", False) # Flag passato dal config model
    
    if mode == "families" and use_weights:
        print("Computing Class Weights...")
        labels = train_df['family_label'].values
        classes = np.unique(labels)
        
        # Calcolo pesi 'balanced' di sklearn
        cw = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
        
        # Conversione a tensore
        # Nota: PyTorch vuole FloatTensor
        class_weights_tensor = torch.tensor(cw, dtype=torch.float)
        
        # Stampa pesi per debug
        for cls, weight in zip(classes, cw):
            print(f"  Class {cls}: {weight:.4f}")

    # Dataset Creation
    # Passiamo is_train=True al training set per attivare l'augmentation
    train_ds = CodeDataset(
        train_df, 
        tokenizer, 
        common_cfg["max_length"], 
        mode=mode, 
        is_train=True 
    )
    
    val_ds = CodeDataset(
        val_df, 
        tokenizer, 
        common_cfg["max_length"], 
        mode=mode, 
        is_train=False
    )

    return train_ds, val_ds, class_weights_tensor