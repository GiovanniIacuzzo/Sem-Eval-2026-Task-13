import torch
import pandas as pd
import logging
import math
import random
import os
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, is_train=False, aug_config=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        # Configurazione Data Augmentation
        self.aug_config = aug_config if aug_config else {}
        self.aug_prob = self.aug_config.get("prob", 0.7)
        self.newline_prob = self.aug_config.get("newline_prob", 0.5)
        self.comment_drop_prob = self.aug_config.get("comment_drop_prob", 0.3)
        
        # Pulizia preventiva
        dataframe = dataframe.dropna(subset=['code', 'label'])
        
        self.codes = dataframe['code'].astype(str).tolist()
        self.labels = dataframe['label'].astype(int).tolist()
        
        # Log iniziale
        if is_train:
            logger.info(f"[Dataset] Initialized Train Set. Augmentation Enabled: {self.aug_prob > 0}")

    def _augment_code(self, code):
        """
        Data Augmentation parametrica per simulare OOD.
        """
        # 1. Skip casuale
        if random.random() > self.aug_prob:
            return code
            
        lines = code.split('\n')
        
        # 2. Inserimento Random Newlines
        if random.random() < self.newline_prob:
            if len(lines) > 2:
                idx = random.randint(0, len(lines)-1)
                lines.insert(idx, "")
        
        # 3. Rimozione commenti casuale
        if random.random() < self.comment_drop_prob:
            lines = [l for l in lines if not l.strip().startswith(('#', '//'))]
            
        return '\n'.join(lines)

    def _extract_features(self, code):
        """
        Estrae feature stilistiche NORMALIZZATE per evitare esplosioni.
        Usa np.log1p per schiacciare i valori grandi.
        """
        features = []
        l_code = len(code) + 1.0
        
        # 1. Log Length
        features.append(math.log1p(l_code)) 
        
        # 2. Ratio Spazi
        features.append(code.count(' ') / l_code)
        
        # 3. Ratio Newlines
        features.append(code.count('\n') / l_code)
        
        # 4. Ratio Simboli Speciali
        symbols = sum(code.count(c) for c in "{}[];,")
        features.append(math.log1p(symbols) / math.log1p(l_code))
        
        # 5. Comment Density
        comments = code.count('#') + code.count('//')
        features.append(comments / (code.count('\n') + 1.0))

        return torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        code = self.codes[idx]
        label = self.labels[idx]
        
        # Data Augmentation solo in training
        if self.is_train:
            code = self._augment_code(code)
            
        # Tokenizzazione UniXcoder
        inputs = self.tokenizer(
            code,
            return_tensors=None, 
            max_length=self.max_length,
            truncation=True,
            padding=False 
        )
        
        # Feature manuali
        extra = self._extract_features(code)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": label,
            "extra_features": extra
        }

def load_data(config, tokenizer):
    """
    Carica e BILANCIA i dati usando i parametri dal config.
    Versione Robustezza OOD + Debugging.
    """
    data_dir = config["data_dir"]
    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "validation.parquet")

    logger.info(f"Loading raw data from {data_dir}...")
    try:
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
    except Exception as e:
        logger.error(f"Errore caricamento dati: {e}")
        raise e

    logger.info(f"Train Rows Base: {len(train_df)}")
    if 'language' in train_df.columns:
        logger.info(f"Languages found in raw data: {train_df['language'].unique()}")
        
        # 1. NORMALIZZAZIONE
        train_df['language'] = train_df['language'].astype(str).str.lower().str.strip()
    else:
        logger.error("CRITICAL: Colonna 'language' non trovata nel dataframe!")
        raise KeyError("Column 'language' missing")

    # --- STRATEGIA DI BILANCIAMENTO LINGUAGGI ---
    if config.get("balance_languages", True):
        target_size = config.get("target_size_per_language", 35000)
        logger.info(f"Applying Language Balancing (Target per lang: {target_size})...")
        
        # Separiamo i linguaggi
        df_py = train_df[train_df['language'] == 'python']
        df_cpp = train_df[train_df['language'] == 'c++']
        df_java = train_df[train_df['language'] == 'java']
        
        logger.info(f"Counts before balance -> Py: {len(df_py)}, C++: {len(df_cpp)}, Java: {len(df_java)}")
        
        if len(df_py) == 0 and len(df_cpp) == 0:
             logger.warning("ATTENZIONE: Non ho trovato né Python né C++. Controlla i nomi dei linguaggi stampati sopra.")

        # Downsampling/Upsampling logica
        # Python: prendiamo un sample
        if len(df_py) > target_size:
            df_py_sample = df_py.sample(n=target_size, random_state=42)
        else:
            df_py_sample = df_py
        
        # C++ e Java: prendiamo tutto
        df_cpp_sample = df_cpp 
        df_java_sample = df_java 
        
        # Ricombiniamo
        train_df = pd.concat([df_py_sample, df_cpp_sample, df_java_sample])
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Balanced Train Size: {len(train_df)} rows")
        if len(train_df) == 0:
            raise ValueError("Il Dataset di training è VUOTO dopo il bilanciamento. Controlla i nomi dei linguaggi.")
            
        logger.info(f"New Language Dist:\n{train_df['language'].value_counts()}")

    # Estrazione Configurazione Augmentation
    aug_config = config.get("augmentation", {})

    train_ds = CodeDataset(
        train_df, 
        tokenizer, 
        max_length=config.get("max_length", 512), 
        is_train=True,
        aug_config=aug_config
    )
    
    val_ds = CodeDataset(
        val_df, 
        tokenizer, 
        max_length=config.get("max_length", 512), 
        is_train=False,
        aug_config=None 
    )
    
    return train_ds, val_ds, None