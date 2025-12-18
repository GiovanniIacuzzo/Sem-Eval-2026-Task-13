import os
import pandas as pd
import numpy as np
import logging

# Configurazione Log
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# CONFIGURAZIONE PERCORSI
DATA_DIR = "data/Task_B"
OUTPUT_DIR = "data/Task_B_Processed"
TRAIN_FILE = "train.parquet"
VAL_FILE = "validation.parquet"

def clean_data(df):
    """
    Pulizia preliminare comune a train e val.
    Prepara le colonne per DANN e Labeling.
    """
    # 1. Normalizzazione Generatore
    if 'generator' in df.columns:
        df['generator'] = df['generator'].astype(str).str.lower().str.strip()
    
    # 2. Normalizzazione Linguaggio (CRITICO PER DANN)
    # Riempiamo i null con 'unknown' e portiamo tutto a lowercase
    if 'language' in df.columns:
        df['language'] = df['language'].fillna('unknown').astype(str).str.lower().str.strip()
    else:
        logger.warning("Colonna 'language' non trovata! Il DANN fallirà.")
        df['language'] = 'unknown'

    # 3. Deduzione Label (Safety Check)
    if 'label' not in df.columns:
        logger.warning("'label' column not found. Creating placeholder...")
        # Logica fallback se necessario, ma per Task B assumiamo label presenti
        df['label'] = -1 
        
    return df

def process_binary_dataset(df, is_train=True):
    """
    Crea il dataset per il GATEKEEPER (Human vs AI).
    Target: 'is_ai' (0=Human, 1=AI)
    """
    df = df.copy()
    
    # 0 = Human, 1-10 = AI -> Convertiamo in 0 e 1
    df['is_ai'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    
    if is_train:
        # BILANCIAMENTO (Solo per il Training)
        # Il Gatekeeper deve vedere 50% Human e 50% AI per non essere biased.
        df_ai = df[df['is_ai'] == 1]
        df_human = df[df['is_ai'] == 0]
        
        n_ai = len(df_ai)
        n_human = len(df_human)
        
        # Undersampling Human (che sono la maggioranza solitamente)
        # Prendiamo un numero di umani pari agli AI
        samples_human = min(n_human, n_ai)
        
        logger.info(f"Binary Balancing -> Original Human: {n_human}, AI: {n_ai}. Keeping {samples_human} Humans.")
        
        df_human_sampled = df_human.sample(n=samples_human, random_state=42)
        
        # Uniamo e mescoliamo
        df_final = pd.concat([df_ai, df_human_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        # Per la validazione NON tocchiamo il bilanciamento. 
        # Dobbiamo valutare sulle proporzioni reali (o quelle fornite dagli organizzatori).
        df_final = df
        
    return df_final[['code', 'is_ai', 'label', 'language', 'generator']]

def process_families_dataset(df, is_train=True):
    """
    Crea il dataset per lo SPECIALIST (Solo AI).
    Target: 'family_label' (0-9)
    """
    df = df.copy()
    
    # FILTRO: Teniamo SOLO le AI. 
    # Human (label 0) viene scartato per questo modello.
    df_ai = df[df['label'] > 0].copy()
    
    if len(df_ai) == 0:
        logger.warning("Nessun dato AI trovato in questo set!")
        return pd.DataFrame()

    # REMAP: Label originali 1..10 -> Nuove label 0..9
    # PyTorch CrossEntropy richiede label che partono da 0.
    df_ai['family_label'] = df_ai['label'] - 1
    
    # Check di integrità
    min_l, max_l = df_ai['family_label'].min(), df_ai['family_label'].max()
    if min_l < 0 or max_l > 9:
        raise ValueError(f"Errore Mapping Famiglie: Label trovate tra {min_l} e {max_l}. Atteso 0-9.")

    if is_train:
        # Mescoliamo il training
        df_ai = df_ai.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Log del mapping per debug
        mapping = df_ai[['label', 'family_label', 'generator']].drop_duplicates().sort_values('family_label')
        logger.info(f"Family Mapping (Original -> Internal):\n{mapping.head(11)}")

    return df_ai[['code', 'family_label', 'label', 'language', 'generator']]

def prepare_datasets():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. PROCESSING TRAINING SET ---
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    if os.path.exists(train_path):
        logger.info(f"Processing TRAIN: {train_path}")
        df_train = pd.read_parquet(train_path)
        df_train = clean_data(df_train)
        
        # A. Binary Train
        df_bin_train = process_binary_dataset(df_train, is_train=True)
        bin_out = os.path.join(OUTPUT_DIR, "train_binary.parquet")
        df_bin_train.to_parquet(bin_out)
        logger.info(f"Saved BINARY Train: {len(df_bin_train)} samples -> {bin_out}")
        
        # B. Families Train
        df_fam_train = process_families_dataset(df_train, is_train=True)
        fam_out = os.path.join(OUTPUT_DIR, "train_families.parquet")
        df_fam_train.to_parquet(fam_out)
        logger.info(f"Saved FAMILIES Train: {len(df_fam_train)} samples -> {fam_out}")
    else:
        logger.error(f"Train file not found: {train_path}")

    # --- 2. PROCESSING VALIDATION SET ---
    # Creiamo dataset di validazione specifici per evitare errori nelle metriche
    val_path = os.path.join(DATA_DIR, VAL_FILE)
    if os.path.exists(val_path):
        logger.info(f"Processing VAL: {val_path}")
        df_val = pd.read_parquet(val_path)
        df_val = clean_data(df_val)
        
        # A. Binary Val (Tutto il validation set, label 0/1)
        df_bin_val = process_binary_dataset(df_val, is_train=False)
        bin_val_out = os.path.join(OUTPUT_DIR, "val_binary.parquet")
        df_bin_val.to_parquet(bin_val_out)
        logger.info(f"Saved BINARY Val: {len(df_bin_val)} samples")
        
        # B. Families Val (SOLO AI, label 0-9)
        # Importante: Non possiamo validare lo specialist su codice umano!
        df_fam_val = process_families_dataset(df_val, is_train=False)
        fam_val_out = os.path.join(OUTPUT_DIR, "val_families.parquet")
        df_fam_val.to_parquet(fam_val_out)
        logger.info(f"Saved FAMILIES Val: {len(df_fam_val)} samples")
    else:
        logger.warning(f"Validation file not found: {val_path}. Skipping val split generation.")

if __name__ == "__main__":
    prepare_datasets()