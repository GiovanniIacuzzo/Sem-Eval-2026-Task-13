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

FAMILY_MAPPING = {
    'deepseek': 0, 'gemma': 1, 'gpt': 2, 'granite': 3, 'llama': 4, 
    'mistral': 5, 'phi': 6, 'qwen': 7, 'starcoder': 8, 'yi': 9, 'other': 10
}

def get_family_name(generator_str):
    gen = str(generator_str).lower().strip()
    
    # --- FIX CRITICA: CATCH HUMAN ---
    if 'human' in gen: return 'human' 
    # --------------------------------
    
    # Ordine importante: controlliamo stringhe specifiche prima di quelle generiche
    if 'granite' in gen or 'ibm' in gen: return 'granite'
    if 'llama' in gen: return 'llama'
    if 'gpt' in gen or 'openai' in gen: return 'gpt'
    if 'mistral' in gen or 'codestral' in gen: return 'mistral'
    if 'qwen' in gen: return 'qwen'
    if 'phi' in gen: return 'phi'
    if 'deepseek' in gen: return 'deepseek'
    if 'gemma' in gen: return 'gemma'
    if 'yi' in gen and 'yi-' in gen: return 'yi'
    if 'starcoder' in gen or 'bigcode' in gen or 'santa' in gen: return 'starcoder'
    
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
    # Usiamo la label originale del dataset (che è sicura) per il binary
    df['is_ai'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    return df[['code', 'is_ai', 'label', 'language', 'family', 'generator']]

def process_families_dataset(df, mapping_file=None, is_train=True):
    df = df.copy()
    
    if 'family' not in df.columns:
        df['family'] = df['generator'].apply(get_family_name)
        
    # ORA QUESTO FUNZIONA CORRETTAMENTE
    # Rimuove 'human' e lascia solo le AI
    df_ai = df[df['family'] != 'human'].copy() 
    
    # Mapping: se qualcosa non è nel dizionario ma non è human, diventa 'other'
    df_ai['family_norm'] = df_ai['family'].apply(lambda x: x if x in FAMILY_MAPPING else 'other')
    df_ai['family_label'] = df_ai['family_norm'].map(FAMILY_MAPPING)
    
    logger.info(f"Original size: {len(df)} -> Filtered AI Only: {len(df_ai)}")
    logger.info(f"Class counts:\n{df_ai['family_norm'].value_counts()}")
    
    return df_ai[['code', 'family_label', 'family_norm']]

def prepare_datasets():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping_path = os.path.join(OUTPUT_DIR, "family_mapping.json")
    
    # Salva il mapping per riferimento futuro
    with open(mapping_path, 'w') as f:
        json.dump(FAMILY_MAPPING, f, indent=4)
    
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    if os.path.exists(train_path):
        logger.info("Processing TRAIN...")
        df_train = clean_data(pd.read_parquet(train_path))
        
        df_bin_train = process_binary_dataset(df_train, is_train=True)
        df_bin_train.to_parquet(os.path.join(OUTPUT_DIR, "train_binary.parquet"))
        
        df_fam_train = process_families_dataset(df_train, mapping_file=mapping_path, is_train=True)
        df_fam_train.to_parquet(os.path.join(OUTPUT_DIR, "train_families.parquet"))

    val_path = os.path.join(DATA_DIR, VAL_FILE)
    if os.path.exists(val_path):
        logger.info("Processing VAL...")
        df_val = clean_data(pd.read_parquet(val_path))
        
        df_bin_val = process_binary_dataset(df_val, is_train=False)
        df_bin_val.to_parquet(os.path.join(OUTPUT_DIR, "val_binary.parquet"))
        
        df_fam_val = process_families_dataset(df_val, mapping_file=mapping_path, is_train=False)
        df_fam_val.to_parquet(os.path.join(OUTPUT_DIR, "val_families.parquet"))

if __name__ == "__main__":
    prepare_datasets()