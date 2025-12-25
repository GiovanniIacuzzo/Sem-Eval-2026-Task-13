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
IMG_PATH = os.getenv("IMG_PATH", "img/img_TaskB")

# Ensure output directory exists
os.makedirs(IMG_PATH, exist_ok=True)

FILES = {
    "Train": "train.parquet",
    "Validation": "validation.parquet",
    "Test": "test_sample.parquet"
}

# -----------------------------------------------------------------------------
# Label Normalization Logic
# -----------------------------------------------------------------------------
def map_to_family(model_name: str) -> str:
    """
    Mappa le 31 varianti specifiche in 11 famiglie di modelli principali.
    Logica basata sulla presenza di sottostringhe.
    """
    if pd.isna(model_name):
        return "unknown"
        
    name = str(model_name).lower().strip()

    if 'llama' in name: return 'llama'
    if 'gpt' in name: return 'gpt'
    if 'mistral' in name or 'mixtral' in name: return 'mistral'
    if 'gemma' in name: return 'gemma'
    if 'claude' in name: return 'claude'
    if 'qwen' in name: return 'qwen'
    if 'phi' in name: return 'phi'
    if 'deepseek' in name: return 'deepseek'
    if 'starcoder' in name: return 'starcoder'
    if 'yi' in name: return 'yi'
    if 'gemini' in name: return 'gemini'
    
    return name

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
        
        print(f"   -> Mapping generators (original unique: {df['generator'].nunique()})...")
        df['generator'] = df['generator'].apply(map_to_family)
        print(f"   -> Mapped to families (new unique: {df['generator'].nunique()})")
    
    # 2. Feature Engineering
    if 'code' in df.columns:
        df['char_length'] = df['code'].str.len()
        df['line_count'] = df['code'].str.count('\n') + 1
        df['token_count_approx'] = df['code'].apply(lambda x: len(str(x).split()))

        # Rimuovi snippet vuoti
        initial_len = len(df)
        df = df[df['char_length'] > 0].copy()
        if len(df) < initial_len:
            print(f"   -> Removed {initial_len - len(df)} empty rows.")

        df['code_preview'] = df['code'].str[:100].str.replace('\n', '\\n')
    
    return df

# -----------------------------------------------------------------------------
# Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------------------
def eda_dataset(df: pd.DataFrame, dataset_name: str):
    if df is None: return

    print(f"\n{'='*60}\nEDA REPORT: {dataset_name}\n{'='*60}")
    print(f"Total Samples: {len(df)}")
    
    if 'code' in df.columns:
        num_dupes = df.duplicated(subset=['code']).sum()
        if num_dupes > 0:
            print(f"[WARNING] Found {num_dupes} duplicate snippets.")

    # 2. Class Distribution
    if 'generator' in df.columns:
        print("\n--- Class Distribution (Families) ---")
        class_counts = df['generator'].value_counts()
        print(class_counts)
        print(f"Total Labels Found: {len(class_counts)}")
        
        if len(class_counts) != 11 and dataset_name != "Test":
            print(f"[NOTE] Attenzione: Trovate {len(class_counts)} label invece di 11.")

    # ---------------------------------------
    # Visualizations
    # ---------------------------------------
    sns.set_theme(style="whitegrid")
    
    # Plot A: Class Balance (Aggiornato per le famiglie)
    if 'generator' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='generator', order=df['generator'].value_counts().index, palette="viridis", hue='generator', legend=False)
        plt.title(f"Model Family Distribution - {dataset_name}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_PATH, f"{dataset_name}_class_dist.png"))
        plt.close()

    # Plot B: Token Length vs Generator Family
    if 'generator' in df.columns:
        plt.figure(figsize=(14, 8))
        limit = df['token_count_approx'].quantile(0.95)
        viz_df = df[df['token_count_approx'] < limit]
        
        order = viz_df.groupby('generator')['token_count_approx'].median().sort_values(ascending=False).index
        
        sns.boxplot(data=viz_df, x='generator', y='token_count_approx', order=order, palette="coolwarm", hue='generator', legend=False)
        plt.title(f"Token Count by Model Family - {dataset_name}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_PATH, f"{dataset_name}_token_boxplot.png"))
        plt.close()

    # Plot C: Heatmap
    if 'generator' in df.columns and 'language' in df.columns:
        plt.figure(figsize=(12, 8))
        top_langs = df['language'].value_counts().head(10).index
        filtered_df = df[df['language'].isin(top_langs)]
        
        crosstab = pd.crosstab(filtered_df['generator'], filtered_df['language'], normalize='index')
        sns.heatmap(crosstab, annot=True, fmt='.2f', cmap="Blues", cbar_kws={'label': 'Probability'})
        plt.title(f"Model Family vs Language Probability - {dataset_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_PATH, f"{dataset_name}_heatmap_norm.png"))
        plt.close()

    print(f"Images saved to {IMG_PATH}")
    
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
    train_df = load_and_preprocess(FILES["Train"])
    eda_dataset(train_df, "Train")

    val_df = load_and_preprocess(FILES["Validation"])
    eda_dataset(val_df, "Validation")

    test_df = load_and_preprocess(FILES["Test"])
    eda_dataset(test_df, "Test")