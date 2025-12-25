import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------
# Path
# ---------------------------------------
data_path = os.getenv("DATA_PATH")
img_path = os.getenv("IMG_PATH", "./img/img_TaskA")
os.makedirs(img_path, exist_ok=True)

files = {
    "Train": "Task_A/train.parquet",
    "Validation": "Task_A/validation.parquet",
    "Test": "Task_A/test_sample.parquet"
}

# ---------------------------------------
# Funzione di caricamento e preprocessing
# ---------------------------------------
def load_and_preprocess(file_path: str):
    df = pd.read_parquet(file_path)
    
    # Uniforma stringhe
    df['language'] = df['language'].str.lower()
    df['generator'] = df['generator'].str.lower()
    
    # Calcola lunghezza codice
    df['code_length'] = df['code'].str.len()
    
    # Rimuove snippet vuoti
    df = df[df['code_length'] > 0].copy()
    
    # Troncamento snippet troppo lunghi
    MAX_LEN = 2048
    df['code'] = df['code'].str[:MAX_LEN]
    
    return df

# ---------------------------------------
# Funzione EDA avanzata
# ---------------------------------------
def eda_dataset(df: pd.DataFrame, name: str):
    print(f"\n{'='*60}\nEDA: {name}\n{'='*60}\n")
    
    print("Prime righe (codice troncato a 200 char):")
    print(df.head().assign(code=lambda x: x['code'].str[:200]))
    
    print("\nInfo generali:")
    print(df.info())
    
    print(f"\nNumero righe: {len(df)}")
    
    # Distribuzione label
    print("\nDistribuzione label:")
    print(df['label'].value_counts())
    
    # Distribuzione linguaggi
    print("\nDistribuzione linguaggi:")
    print(df['language'].value_counts())
    
    # Lunghezza codice
    print("\nStatistiche lunghezza codice:")
    print(df['code_length'].describe())
    
    # ---------------------------------------
    # Visualizzazioni
    # ---------------------------------------
    # 1. Istogramma lunghezza snippet per label
    plt.figure(figsize=(12,5))
    sns.histplot(df, x='code_length', hue='label', bins=100, log_scale=True)
    plt.title(f"Lunghezza snippet per label - {name}")
    plt.xlabel("Lunghezza codice (char)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(img_path, f"{name}_length_label.png"))
    plt.close()

    # 2. Distribuzione label per linguaggio
    plt.figure(figsize=(12,5))
    sns.countplot(data=df, x='language', hue='label')
    plt.title(f"Distribuzione label per linguaggio - {name}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(img_path, f"{name}_label_language.png"))
    plt.close()
    
    # 3. Top 10 generatori
    plt.figure(figsize=(12,5))
    top_gen = df['generator'].value_counts().head(10)
    sns.barplot(x=top_gen.values, y=top_gen.index)
    plt.title(f"Top 10 generatori - {name}")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(img_path, f"{name}_top_generators.png"))
    plt.close()

# ---------------------------------------
# Caricamento e EDA
# ---------------------------------------
train_df = load_and_preprocess(os.path.join(data_path, files["Train"]))
dev_df   = load_and_preprocess(os.path.join(data_path, files["Validation"]))
test_df  = load_and_preprocess(os.path.join(data_path, files["Test"]))

eda_dataset(train_df, "Train")
eda_dataset(dev_df, "Validation")
eda_dataset(test_df, "Test")

print(f"Tutte le immagini sono state salvate nella cartella {img_path}")