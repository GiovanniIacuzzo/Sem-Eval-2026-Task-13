import os
import pandas as pd
import logging
import json

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

DATA_DIR = "data/Task_C"
OUTPUT_DIR = "data/Task_C_Processed"
TRAIN_FILE = "train.parquet"
VAL_FILE = "validation.parquet"

LABEL_MAPPING = {
    0: "Human",
    1: "AI-Generated",
    2: "Hybrid",
    3: "Adversarial"
}

def clean_data(df):
    """
    Pulizia base del dataframe:
    - Rimuove righe senza codice
    - Converte le label in int
    - Normalizza i nomi dei linguaggi
    """
    df = df.copy()
    
    initial_len = len(df)
    df = df.dropna(subset=['code'])
    df = df[df['code'].str.strip().astype(bool)]
    
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows due to empty code.")

    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
    
    valid_mask = df['label'].isin([0, 1, 2, 3])
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} rows with invalid labels. Dropping them.")
        df = df[valid_mask]

    if 'language' in df.columns:
        df['language'] = df['language'].fillna('unknown').astype(str).str.lower().str.strip()
    
    return df

def process_dataset(df, dataset_type="Train"):
    """
    Elabora il dataset per il Task C.
    Aggiunge metadati utili e stampa le statistiche.
    """
    df = df.copy()
    
    df['label_name'] = df['label'].map(LABEL_MAPPING)
    df['is_generated'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    
    # Log statistiche
    logger.info(f"--- {dataset_type} Statistics ---")
    logger.info(f"Total Samples: {len(df)}")
    
    counts = df['label'].value_counts().sort_index()
    for label_id, count in counts.items():
        name = LABEL_MAPPING.get(label_id, "Unknown")
        percentage = (count / len(df)) * 100
        logger.info(f"Class {label_id} ({name}): {count} ({percentage:.2f}%)")
        
    return df[['code', 'label', 'label_name', 'is_generated', 'language']]

def prepare_datasets():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Salva il mapping JSON per riferimento
    mapping_path = os.path.join(OUTPUT_DIR, "label_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(LABEL_MAPPING, f, indent=4)
    
    # --- PROCESSING TRAIN ---
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    if os.path.exists(train_path):
        logger.info(f"Reading TRAIN from {train_path}...")
        df_train = pd.read_parquet(train_path)
        
        # Clean
        df_train = clean_data(df_train)
        
        # Process
        df_train_proc = process_dataset(df_train, dataset_type="TRAIN")
        
        # Save
        save_path = os.path.join(OUTPUT_DIR, "train_processed.parquet")
        df_train_proc.to_parquet(save_path)
        logger.info(f"Saved processed TRAIN to {save_path}\n")
    else:
        logger.error(f"Train file not found at {train_path}")

    # --- PROCESSING VALIDATION ---
    val_path = os.path.join(DATA_DIR, VAL_FILE)
    if os.path.exists(val_path):
        logger.info(f"Reading VALIDATION from {val_path}...")
        df_val = pd.read_parquet(val_path)
        
        # Clean
        df_val = clean_data(df_val)
        
        # Process
        df_val_proc = process_dataset(df_val, dataset_type="VALIDATION")
        
        # Save
        save_path = os.path.join(OUTPUT_DIR, "val_processed.parquet")
        df_val_proc.to_parquet(save_path)
        logger.info(f"Saved processed VALIDATION to {save_path}")
    else:
        logger.error(f"Validation file not found at {val_path}")

if __name__ == "__main__":
    prepare_datasets()