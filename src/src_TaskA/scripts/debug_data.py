import pandas as pd
import numpy as np
import os
import argparse

pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 50)

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def check_dataset_quality(file_path):
    print_header("DEBUG DATASET OOD (Out-Of-Distribution)")
    
    if not os.path.exists(file_path):
        print(f"ERRORE CRITICO: Il file {file_path} non esiste.")
        print("   Verifica se il processo precedente ha salvato correttamente prima di chiudersi.")
        return

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"ERRORE LETTURA: Impossibile aprire il file parquet.\n   {e}")
        return

    # 1. Statistiche Generali
    print(f"File caricato: {file_path}")
    print(f"Totale Righe: {len(df)}")
    
    if len(df) == 0:
        print("  ATTENZIONE: Il dataset è VUOTO!")
        return

    # 2. Controllo Distribuzione Linguaggi
    print_header("1. DISTRIBUZIONE LINGUAGGI")
    print(df['language'].value_counts())
    
    expected_langs = ["go", "c#", "php", "javascript", "c"]
    found_langs = df['language'].unique()
    missing = [l for l in expected_langs if l not in found_langs]
    
    if missing:
        print(f"\n  MANCANO questi linguaggi target: {missing}")
    else:
        print("\nOttimo! Tutti i linguaggi target sono presenti.")

    # 3. Controllo Label
    print_header("2. CONTROLLO LABEL")
    label_counts = df['label'].value_counts()
    print(label_counts)
    
    if 0 in label_counts.index:
        print("\nERRORE GRAVE: Trovate label '0' (Umano) in dati generati!")
        print("   Tutte le righe generate dovrebbero avere label=1.")
    else:
        print("\nCorretto: Tutte le righe sono etichettate come 1 (Machine).")

    # 4. Analisi Lunghezza Codice
    print_header("3. STATISTICHE LUNGHEZZA")
    df['char_len'] = df['code'].str.len()
    print(df['char_len'].describe().round(1))
    
    short_code = df[df['char_len'] < 50]
    if len(short_code) > 0:
        print(f"\n  WARNING: {len(short_code)} snippet sono molto corti (<50 char).")
        print("   Esempio corto:", short_code.iloc[0]['code'])

    # 5. Controllo "Pollution"
    print_header("4. CONTROLLO PULIZIA (CHAT ARTIFACTS)")
    bad_keywords = [
        "here is", "sure,", "translate", "python code", 
        "output:", "```", "<|im_end|>", "system prompt"
    ]
    
    pollution_stats = {}
    for kw in bad_keywords:
        count = df['code'].str.lower().str.contains(kw, regex=False).sum()
        if count > 0:
            pollution_stats[kw] = count

    if pollution_stats:
        print("  Trovati residui di testo non-codice:")
        for k, v in pollution_stats.items():
            print(f"   - '{k}': {v} righe")
    else:
        print("Pulizia Eccellente: Nessuna keyword 'chat' comune trovata.")

    # 6. Ispezione Visiva
    print_header("5. ISPEZIONE VISIVA (3 CAMPIONI RANDOM)")
    
    # Prendi campioni da linguaggi diversi se possibile
    langs_to_show = df['language'].unique()[:3]
    
    for lang in langs_to_show:
        subset = df[df['language'] == lang]
        if len(subset) == 0: continue
        
        row = subset.sample(1).iloc[0]
        
        print(f"\n--- [Linguaggio: {lang.upper()}] ---")
        print(f"Source ID: {row.get('original_source_id', 'N/A')}")
        print("-" * 40)
        # Stampa primi 500 caratteri formattati
        code_preview = row['code']
        if len(code_preview) > 1000:
            print(code_preview[:1000] + "\n... [TRONCATO] ...")
        else:
            print(code_preview)
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/Task_A/train_augmented_ood.parquet")
    args = parser.parse_args()
    
    check_dataset_quality(args.path)