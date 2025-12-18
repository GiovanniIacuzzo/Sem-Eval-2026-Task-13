import os
import pandas as pd
import numpy as np
import logging

# Configurazione Log
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# CONFIGURAZIONE PERCORSI
# Assumiamo di essere eseguiti dalla root del progetto, o aggiusta i path
ORIGINAL_TRAIN_PATH = "data/Task_B/train.parquet"
OUTPUT_DIR = "data/Task_B_Processed"

# Mappa Famiglie (Task 13 B)
# 0 = Human, 1-10 = AI
# Questa mappa serve per essere sicuri al 100% delle assegnazioni
FAMILY_MAP = {
    'human': 0,
    # Classi AI (semplificate per check, la logica si basa sul fatto che human è l'unica 0)
}

def prepare_datasets():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Loading original data from {ORIGINAL_TRAIN_PATH}...")
    try:
        df = pd.read_parquet(ORIGINAL_TRAIN_PATH)
    except Exception as e:
        logger.error(f"Errore caricamento file. Sei sicuro del percorso? {e}")
        return

    # Normalizzazione stringhe
    if 'generator' in df.columns:
        df['generator'] = df['generator'].astype(str).str.lower().str.strip()
    
    # 1. Creiamo la colonna 'is_ai' (0 = Human, 1 = AI)
    # Assumiamo che la colonna 'label' esista già e sia corretta (0-10).
    # Se non esiste, la deduciamo dal generator.
    if 'label' not in df.columns:
        logger.warning("'label' column not found. Creating it from 'generator'...")
        # Qui dovresti usare la tua FAMILY_MAP completa se il parquet non ha le label
        df['label'] = df['generator'].apply(lambda x: 0 if x == 'human' else -1) 
        # Nota: Se il parquet non ha label numeriche, questo script si ferma qui. 
        # Si presume che il train.parquet abbia già le label corrette del task.

    df['is_ai'] = df['label'].apply(lambda x: 1 if x > 0 else 0)

    # ---------------------------------------------------------
    # DATASET A: BINARY (Gatekeeper)
    # Obiettivo: 50% Human, 50% AI. 
    # ---------------------------------------------------------
    logger.info("Creating Binary Dataset (Balanced Human/AI)...")
    
    df_ai = df[df['is_ai'] == 1]
    df_human = df[df['is_ai'] == 0]
    
    n_ai = len(df_ai)
    n_human = len(df_human)
    
    logger.info(f"Original counts -> AI: {n_ai}, Human: {n_human}")
    
    # Undersampling Human: Ne prendiamo tanti quanti sono gli AI (o poco più)
    # Mettiamo un moltiplicatore 1.0 per fare 50/50 perfetto
    samples_human = min(n_human, n_ai) 
    
    df_human_sampled = df_human.sample(n=samples_human, random_state=42)
    
    df_binary = pd.concat([df_ai, df_human_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Salviamo solo le colonne utili
    cols_to_keep = ['code', 'is_ai', 'label', 'language', 'generator'] # Teniamo label originale per debug
    df_binary = df_binary[cols_to_keep]
    
    binary_out = os.path.join(OUTPUT_DIR, "train_binary.parquet")
    df_binary.to_parquet(binary_out)
    logger.info(f"Saved Binary Train: {len(df_binary)} samples -> {binary_out}")

    # ---------------------------------------------------------
    # DATASET B: FAMILIES (Specialist)
    # Obiettivo: Solo AI. Label rimappate da 0 a 9.
    # ---------------------------------------------------------
    logger.info("Creating Families Dataset (AI Only)...")
    
    df_families = df_ai.copy().sample(frac=1, random_state=42).reset_index(drop=True)
    
    # CRITICO: PyTorch CrossEntropy vuole label 0...N-1.
    # Le tue label AI sono 1...10.
    # Dobbiamo fare label - 1.
    # Quindi: 01-ai (che era 1) diventa 0. OpenAI (che era 10) diventa 9.
    df_families['family_label'] = df_families['label'] - 1
    
    # Controllo sicurezza
    assert df_families['family_label'].min() >= 0, "Errore: Trovata label negativa!"
    assert df_families['family_label'].max() <= 9, "Errore: Trovata label > 9!"

    families_out = os.path.join(OUTPUT_DIR, "train_families.parquet")
    df_families.to_parquet(families_out)
    logger.info(f"Saved Families Train: {len(df_families)} samples -> {families_out}")

if __name__ == "__main__":
    prepare_datasets()