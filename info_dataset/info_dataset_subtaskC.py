import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()

# Base paths
BASE_DATA_PATH = os.getenv("DATA_PATH", "./data")
TASK_C_DIR = os.path.join(BASE_DATA_PATH, "Task_C")
IMG_PATH = os.getenv("IMG_PATH", "./img_TaskC")

# Ensure output directory exists
os.makedirs(IMG_PATH, exist_ok=True)

# File configuration for Task C
FILES = {
    "Train": "train.parquet",
    "Validation": "validation.parquet",
    "Test": "test.parquet"
}

# -----------------------------------------------------------------------------
# Data Loading & Preprocessing
# -----------------------------------------------------------------------------
def load_and_preprocess(file_name: str) -> pd.DataFrame:
    file_path = os.path.join(TASK_C_DIR, file_name)
    
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        return None

    print(f"Loading {file_name}...")
    df = pd.read_parquet(file_path)
    
    # Text normalization
    if 'language' in df.columns:
        df['language'] = df['language'].str.lower()
        
    # Calcolo feature di base
    if 'code' in df.columns:
        df['code_length'] = df['code'].str.len()
        # Preview per display
        df['code_preview'] = df['code'].str[:200]
        
    # Gestione specifica Task C:
    # Se esiste una colonna 'changes' o 'diff' (tipica di task di refactoring/mixed)
    if 'diff' in df.columns:
        df['diff_length'] = df['diff'].str.len()

    return df

# -----------------------------------------------------------------------------
# Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------------------
def eda_dataset(df: pd.DataFrame, dataset_name: str):
    if df is None: return

    print(f"\n{'='*60}\nEDA REPORT: {dataset_name}\n{'='*60}")
    
    # 1. General Stats
    print(f"Total Samples: {len(df)}")
    print("\n--- Columns ---")
    print(df.columns.tolist())
    
    print("\n--- Data Sample ---")
    # Mostriamo colonne rilevanti dinamiche
    cols_to_show = [c for c in ['label', 'score', 'generator', 'language', 'code_preview'] if c in df.columns]
    print(df[cols_to_show].head())
    
    # Identifica la colonna target probabile (adatta se il nome cambia)
    target_col = 'label' if 'label' in df.columns else ('score' if 'score' in df.columns else None)
    
    # ---------------------------------------
    # Visualizzazioni
    # ---------------------------------------
    sns.set_theme(style="whitegrid")

    # A. Analisi del Target (Dinamica: Categorica o Numerica)
    if target_col:
        plt.figure(figsize=(12, 6))
        
        # Se il target è numerico (es. regressione, soft label) -> Istogramma/KDE
        if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 20:
            sns.histplot(df[target_col], kde=True, bins=30, color='purple')
            plt.title(f"Target Distribution ({target_col}) - {dataset_name}")
        
        # Se il target è categorico (es. classe) -> Countplot
        else:
            order = df[target_col].value_counts().index
            sns.countplot(data=df, x=target_col, order=order, palette="viridis", hue=target_col, legend=False)
            plt.title(f"Class Distribution ({target_col}) - {dataset_name}")
            
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_PATH, f"{dataset_name}_target_dist.png"))
        plt.close()
        
        print(f"\n--- Target Distribution ({target_col}) ---")
        if df[target_col].nunique() < 50:
             print(df[target_col].value_counts())
        else:
             print(df[target_col].describe())

    # B. Analisi Lunghezza Codice
    if 'code_length' in df.columns:
        plt.figure(figsize=(12, 6))
        # Filtro estremi per leggibilità
        viz_df = df[df['code_length'] < df['code_length'].quantile(0.95)]
        
        if target_col and df[target_col].nunique() < 20:
            sns.boxplot(data=viz_df, x=target_col, y='code_length', palette="coolwarm", hue=target_col, legend=False)
        else:
            sns.histplot(viz_df['code_length'], bins=50, color="teal")
            
        plt.title(f"Code Length Distribution (95th percentile) - {dataset_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_PATH, f"{dataset_name}_length_dist.png"))
        plt.close()

    # C. Analisi Linguaggi (Top 15)
    if 'language' in df.columns:
        plt.figure(figsize=(12, 6))
        top_langs = df['language'].value_counts().head(15)
        sns.barplot(x=top_langs.values, y=top_langs.index, palette="mako", hue=top_langs.index, legend=False)
        plt.title(f"Top 15 Languages - {dataset_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_PATH, f"{dataset_name}_languages.png"))
        plt.close()

    print(f"Images saved to {IMG_PATH}")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Load Train
    train_df = load_and_preprocess(FILES["Train"])
    eda_dataset(train_df, "Train")

    # Load Validation
    val_df = load_and_preprocess(FILES["Validation"])
    eda_dataset(val_df, "Validation")

    # Load Test (Sample) if needed
    test_df = load_and_preprocess(FILES.get("Test", "test_sample.parquet"))
    eda_dataset(test_df, "Test_Sample")