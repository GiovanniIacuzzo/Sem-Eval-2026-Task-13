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

# --- FAMIGLIE ---
FAMILY_GROUPS = {
    'gpt': 'gpt',
    'llama': 'llama',
    'qwen': 'qwen',
    'phi': 'phi',
    'mistral': 'mistral',
    'granite': 'granite',
    'deepseek': 'deepseek',
    'starcoder': 'starcoder',
    'gemma': 'gemma',
    'yi': 'yi',
    'codestral': 'mistral',
    'claude': 'anthropic',
}

def get_family_name(generator_str):
    """Estrae la famiglia dal nome del generatore."""
    gen = str(generator_str).lower()
    for key, family in FAMILY_GROUPS.items():
        if key in gen:
            return family
    return 'other'

def clean_data(df):
    df = df.copy()
    if 'generator' in df.columns:
        df['generator_clean'] = df['generator'].astype(str).str.lower().str.strip()
        df['family'] = df['generator_clean'].apply(get_family_name)
    
    if 'language' in df.columns:
        df['language'] = df['language'].fillna('unknown').astype(str).str.lower().str.strip()
    
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
    return df

def process_binary_dataset(df, is_train=True):
    df = df.copy()
    df['is_ai'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    return df[['code', 'is_ai', 'label', 'language', 'family', 'generator']]

def process_families_dataset(df, mapping_file=None, is_train=True):
    """
    Crea il dataset mappando i generatori in MACRO-FAMIGLIE.
    """
    df = df.copy()
    df_ai = df[df['label'] > 0].copy()
    
    if len(df_ai) == 0:
        return pd.DataFrame()

    if is_train:
        unique_families = sorted(df_ai['family'].unique())
        family_map = {fam: i for i, fam in enumerate(unique_families)}
        
        if mapping_file:
            with open(mapping_file, 'w') as f:
                json.dump(family_map, f, indent=4)
            logger.info(f"Saved Global Family Mapping: {family_map}")
    else:
        if mapping_file and os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                family_map = json.load(f)
        else:
            unique_families = sorted(df_ai['family'].unique())
            family_map = {fam: i for i, fam in enumerate(unique_families)}

    df_ai['family_label'] = df_ai['family'].map(family_map).fillna(family_map.get('other', 0)).astype(int)

    if is_train:
        df_ai = df_ai.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_ai[['code', 'family_label', 'family', 'language', 'generator']]

def prepare_datasets():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping_path = os.path.join(OUTPUT_DIR, "family_mapping.json")
    
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    if os.path.exists(train_path):
        df_train = clean_data(pd.read_parquet(train_path))
        
        df_bin_train = process_binary_dataset(df_train, is_train=True)
        df_bin_train.to_parquet(os.path.join(OUTPUT_DIR, "train_binary.parquet"))
        
        df_fam_train = process_families_dataset(df_train, mapping_file=mapping_path, is_train=True)
        df_fam_train.to_parquet(os.path.join(OUTPUT_DIR, "train_families.parquet"))
        logger.info(f"Train processed: {len(df_fam_train)} AI samples in families.")

    val_path = os.path.join(DATA_DIR, VAL_FILE)
    if os.path.exists(val_path):
        df_val = clean_data(pd.read_parquet(val_path))
        
        df_bin_val = process_binary_dataset(df_val, is_train=False)
        df_bin_val.to_parquet(os.path.join(OUTPUT_DIR, "val_binary.parquet"))
        
        df_fam_val = process_families_dataset(df_val, mapping_file=mapping_path, is_train=False)
        df_fam_val.to_parquet(os.path.join(OUTPUT_DIR, "val_families.parquet"))
        logger.info(f"Val processed: {len(df_fam_val)} AI samples.")

if __name__ == "__main__":
    prepare_datasets()