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
TASK_B_DIR = os.path.join(BASE_DATA_PATH, "Task_B")
IMG_PATH = os.getenv("IMG_PATH", "./img_TaskB")

# Ensure output directory exists
os.makedirs(IMG_PATH, exist_ok=True)

# File configuration
FILES = {
    "Train": "train.parquet",
    "Validation": "validation.parquet",
    "Test": "test.parquet"
}

# -----------------------------------------------------------------------------
# Data Loading & Preprocessing
# -----------------------------------------------------------------------------
def load_and_preprocess(file_name: str) -> pd.DataFrame:
    file_path = os.path.join(TASK_B_DIR, file_name)
    
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        return None

    print(f"Loading {file_name}...")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[ERROR] Could not read parquet file: {e}")
        return None
    
    # 1. Normalizzazione Testo
    if 'language' in df.columns:
        df['language'] = df['language'].str.lower().str.strip()
    
    if 'generator' in df.columns:
        df['generator'] = df['generator'].str.lower().str.strip()
    
    # 2. Feature Engineering Avanzata
    if 'code' in df.columns:
        # Lunghezza in caratteri
        df['char_length'] = df['code'].str.len()
        
        # Lunghezza in linee (utile: alcuni LLM sono più "verticali" di altri)
        df['line_count'] = df['code'].str.count('\n') + 1
        
        # Lunghezza in token (approssimazione splitting per spazi)
        df['token_count_approx'] = df['code'].apply(lambda x: len(str(x).split()))

        # Rimuovi snippet vuoti o nulli
        initial_len = len(df)
        df = df[df['char_length'] > 0].copy()
        if len(df) < initial_len:
            print(f"   -> Removed {initial_len - len(df)} empty rows.")

        # Preview per display
        df['code_preview'] = df['code'].str[:100].str.replace('\n', '\\n')
    
    return df

# -----------------------------------------------------------------------------
# Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------------------
def eda_dataset(df: pd.DataFrame, dataset_name: str):
    if df is None: return

    print(f"\n{'='*60}\nEDA REPORT: {dataset_name}\n{'='*60}")
    
    # 1. General Stats
    print(f"Total Samples: {len(df)}")
    
    # Check duplicati
    if 'code' in df.columns:
        num_dupes = df.duplicated(subset=['code']).sum()
        if num_dupes > 0:
            print(f"\n[WARNING] Found {num_dupes} duplicate code snippets! ({num_dupes/len(df):.2%})")
        else:
            print("\n[OK] No exact duplicate code snippets found.")

    print("\n--- Data Sample ---")
    # Mostra colonne disponibili dinamicamente
    cols_available = [c for c in ['generator', 'language', 'line_count', 'code_preview'] if c in df.columns]
    print(df[cols_available].head())
    
    # 2. Class Distribution (Solo se abbiamo le label)
    if 'generator' in df.columns:
        print("\n--- Class Distribution (Top 10) ---")
        class_counts = df['generator'].value_counts()
        print(class_counts.head(10))
        
        balance_ratio = class_counts.min() / class_counts.max()
        print(f"\nClass Balance Ratio (Min/Max): {balance_ratio:.4f}")
        if balance_ratio < 0.1:
            print("[ALERT] Dataset is heavily imbalanced!")

    # 3. Stats sulle lunghezze
    print("\n--- Length Statistics (Tokens Approx) ---")
    print(df['token_count_approx'].describe())

    # ---------------------------------------
    # Visualizations
    # ---------------------------------------
    sns.set_theme(style="whitegrid")
    palette = "viridis"

    # --- PLOT GENERICI (Funzionano anche per TEST set senza label) ---
    
    # Plot 1: Istogramma distribuzione lunghezza (Generale)
    plt.figure(figsize=(10, 6))
    # Taglio al 95% per leggibilità
    limit = df['token_count_approx'].quantile(0.95)
    viz_df = df[df['token_count_approx'] < limit]
    
    sns.histplot(viz_df['token_count_approx'], bins=50, kde=True, color="teal")
    plt.title(f"Token Count Distribution (General) - {dataset_name}")
    plt.xlabel("Tokens (Approx)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_PATH, f"{dataset_name}_token_dist_general.png"))
    plt.close()

    # --- PLOT SPECIFICI (Solo per TRAIN/VAL con label) ---

    # Plot A: Class Balance
    if 'generator' in df.columns:
        plt.figure(figsize=(12, 6))
        class_counts = df['generator'].value_counts()
        top_classes = class_counts.head(20).index
        sns.countplot(data=df[df['generator'].isin(top_classes)], x='generator', order=top_classes, palette=palette, hue='generator', legend=False)
        plt.title(f"Generator Distribution (Top 20) - {dataset_name}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_PATH, f"{dataset_name}_class_dist.png"))
        plt.close()

    # Plot B: Token Length vs Generator
    if 'generator' in df.columns:
        plt.figure(figsize=(14, 8))
        limit = df['token_count_approx'].quantile(0.95)
        viz_df = df[df['token_count_approx'] < limit]
        order = viz_df.groupby('generator')['token_count_approx'].median().sort_values(ascending=False).index
        
        sns.boxplot(data=viz_df, x='generator', y='token_count_approx', order=order, palette="coolwarm", hue='generator', legend=False)
        plt.title(f"Token Count by Generator - {dataset_name}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_PATH, f"{dataset_name}_token_boxplot.png"))
        plt.close()

    # Plot C: Heatmap
    if 'generator' in df.columns and 'language' in df.columns:
        plt.figure(figsize=(12, 10))
        top_langs = df['language'].value_counts().head(15).index
        filtered_df = df[df['language'].isin(top_langs)]
        crosstab = pd.crosstab(filtered_df['generator'], filtered_df['language'], normalize='index')
        sns.heatmap(crosstab, annot=True, fmt='.2f', cmap="Blues", cbar_kws={'label': 'Probability'})
        plt.title(f"Generator vs Language Probability - {dataset_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_PATH, f"{dataset_name}_heatmap_norm.png"))
        plt.close()

    print(f"Images saved to {IMG_PATH}")
    
    # Helper for Map
    if dataset_name == "Train" and 'generator' in df.columns:
        print("\n" + "="*50)
        print("COPY THIS MAP TO src/dataset/dataset.py")
        print("="*50)
        generators = sorted(df['generator'].unique())
        print("GENERATOR_MAP = {")
        for i, gen in enumerate(generators):
            print(f"    '{gen}': {i},")
        print("}")
        print(f"NUM_LABELS = {len(generators)}")
        print("="*50 + "\n")
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

    # Test Validation
    test_df = load_and_preprocess(FILES["Test"])
    eda_dataset(test_df, "Test")