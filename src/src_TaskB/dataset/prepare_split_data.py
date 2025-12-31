import os
import pandas as pd
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data/Task_B"
OUTPUT_DIR = "data/Task_B_Processed"
TRAIN_FILE = "train.parquet"
VAL_FILE = "validation.parquet"

def clean_data(df):
    """
    Pulizia preliminare comune a train e val.
    """
    if 'generator' in df.columns:
        df['generator'] = df['generator'].astype(str).str.lower().str.strip()
    
    if 'language' in df.columns:
        df['language'] = df['language'].fillna('unknown').astype(str).str.lower().str.strip()
    else:
        logger.warning("Colonna 'language' non trovata! Il DANN fallirà (o userà dummy).")
        df['language'] = 'unknown'

    if 'label' not in df.columns:
        logger.warning("'label' column not found. Creating placeholder -1...")
        df['label'] = -1 
    
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
        
    return df

def process_binary_dataset(df, is_train=True):
    """
    Crea il dataset per il GATEKEEPER (Human vs AI).
    Target: 'is_ai' (0=Human, 1=AI)
    """
    df = df.copy()
    df['is_ai'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    
    if is_train:
        n_total = len(df)
        n_ai = len(df[df['is_ai'] == 1])
        n_human = len(df[df['is_ai'] == 0])
        
        logger.info(f"Binary Dataset Stats -> Human: {n_human}, AI: {n_ai} (Tot: {n_total})")
        
        df_final = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        ratio = n_ai / n_human if n_human > 0 else 0
        logger.info(f"Imbalance Ratio (AI/Human): {ratio:.2f}")
    else:
        df_final = df
        
    return df_final[['code', 'is_ai', 'label', 'language', 'generator']]

def process_families_dataset(df, mapping_file=None, is_train=True):
    """
    Crea il dataset per lo SPECIALIST (Solo AI).
    Target: 'family_label' rimappato da 0 a N-1.
    """
    df = df.copy()
    df_ai = df[df['label'] > 0].copy()
    
    if len(df_ai) == 0:
        logger.warning("Nessun dato AI trovato in questo set!")
        return pd.DataFrame()

    if is_train:
        unique_labels = sorted(df_ai['label'].unique())
        label_map = {orig: new for new, orig in enumerate(unique_labels)}
        
        if mapping_file:
            with open(mapping_file, 'w') as f:
                meta_map = {}
                for orig, new in label_map.items():
                    gen_name = df_ai[df_ai['label'] == orig]['generator'].iloc[0]
                    meta_map[new] = {"original_id": int(orig), "generator": gen_name}
                json.dump(meta_map, f, indent=4)
            logger.info(f"Saved Family Mapping to {mapping_file}")
            
    else:
        if mapping_file and os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                meta_map = json.load(f)
                label_map = {v['original_id']: int(k) for k, v in meta_map.items()}
        else:
             unique_labels = sorted(df_ai['label'].unique())
             label_map = {orig: new for new, orig in enumerate(unique_labels)}

    try:
        df_ai['family_label'] = df_ai['label'].map(label_map)
    except Exception as e:
        logger.error(f"Errore nel mapping delle label: {e}")
        raise e

    if df_ai['family_label'].isnull().any():
        logger.warning("Trovate label nel Validation non presenti nel Train Mapping! Verranno droppate.")
        df_ai = df_ai.dropna(subset=['family_label'])
        df_ai['family_label'] = df_ai['family_label'].astype(int)

    if is_train:
        df_ai = df_ai.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_ai[['code', 'family_label', 'label', 'language', 'generator']]

def prepare_datasets():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping_path = os.path.join(OUTPUT_DIR, "family_mapping.json")
    
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    if os.path.exists(train_path):
        logger.info(f"Processing TRAIN: {train_path}")
        df_train = pd.read_parquet(train_path)
        df_train = clean_data(df_train)
        
        df_bin_train = process_binary_dataset(df_train, is_train=True)
        bin_out = os.path.join(OUTPUT_DIR, "train_binary.parquet")
        df_bin_train.to_parquet(bin_out)
        logger.info(f"Saved BINARY Train: {len(df_bin_train)} samples -> {bin_out}")
        
        df_fam_train = process_families_dataset(df_train, mapping_file=mapping_path, is_train=True)
        fam_out = os.path.join(OUTPUT_DIR, "train_families.parquet")
        df_fam_train.to_parquet(fam_out)
        logger.info(f"Saved FAMILIES Train: {len(df_fam_train)} samples -> {fam_out}")
    else:
        logger.error(f"Train file not found: {train_path}")

    val_path = os.path.join(DATA_DIR, VAL_FILE)
    if os.path.exists(val_path):
        logger.info(f"Processing VAL: {val_path}")
        df_val = pd.read_parquet(val_path)
        df_val = clean_data(df_val)
        
        df_bin_val = process_binary_dataset(df_val, is_train=False)
        bin_val_out = os.path.join(OUTPUT_DIR, "val_binary.parquet")
        df_bin_val.to_parquet(bin_val_out)
        logger.info(f"Saved BINARY Val: {len(df_bin_val)} samples")

        df_fam_val = process_families_dataset(df_val, mapping_file=mapping_path, is_train=False)
        fam_val_out = os.path.join(OUTPUT_DIR, "val_families.parquet")
        df_fam_val.to_parquet(fam_val_out)
        logger.info(f"Saved FAMILIES Val: {len(df_fam_val)} samples")
    else:
        logger.warning(f"Validation file not found: {val_path}. Skipping val split generation.")

if __name__ == "__main__":
    prepare_datasets()