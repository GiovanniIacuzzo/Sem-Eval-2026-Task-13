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
BASE_DATA_PATH = os.getenv("DATA_PATH", "./data") # Default ./data se non settato
TASK_B_DIR = os.path.join(BASE_DATA_PATH, "Task_B")
IMG_PATH = os.getenv("IMG_PATH", "./img/task_b")

# Ensure output directory exists
os.makedirs(IMG_PATH, exist_ok=True)

# File configuration for Task B
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
        print(f"File not found: {file_path}")
        return None

    print(f"Loading {file_name}...")
    df = pd.read_parquet(file_path)
    
    # Text normalization
    if 'language' in df.columns:
        df['language'] = df['language'].str.lower()
    
    # In Task B, 'generator' is the target label
    if 'generator' in df.columns:
        df['generator'] = df['generator'].str.lower()
    
    # Feature Engineering
    if 'code' in df.columns:
        df['code_length'] = df['code'].str.len()
        # Remove empty snippets
        df = df[df['code_length'] > 0].copy()
        # Truncate for analysis speed (optional)
        df['code_preview'] = df['code'].str[:200]
    
    return df

# -----------------------------------------------------------------------------
# Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------------------
def eda_dataset(df: pd.DataFrame, dataset_name: str):
    if df is None: return

    print(f"\n{'='*60}\nEDA REPORT: {dataset_name}\n{'='*60}")
    
    # 1. General Stats
    print(f"Total Samples: {len(df)}")
    print("\n--- Data Sample ---")
    print(df[['generator', 'language', 'code_preview']].head())
    
    # 2. Class Distribution (Generator)
    print("\n--- Class Distribution (Target) ---")
    class_counts = df['generator'].value_counts()
    print(class_counts)
    print(f"Total Unique Classes: {len(class_counts)}")

    # 3. Language Distribution
    print("\n--- Language Distribution ---")
    print(df['language'].value_counts())

    # 4. Length Statistics
    print("\n--- Code Length Statistics ---")
    print(df['code_length'].describe())

    # ---------------------------------------
    # Visualizations
    # ---------------------------------------
    sns.set_theme(style="whitegrid")

    # Plot A: Class Balance (The most important for Multiclass)
    plt.figure(figsize=(14, 6))
    sns.barplot(x=class_counts.values, y=class_counts.index, palette="viridis", hue=class_counts.index, legend=False)
    plt.title(f"Class Distribution (Generators) - {dataset_name}")
    plt.xlabel("Count")
    plt.ylabel("Generator Model")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_PATH, f"{dataset_name}_class_dist.png"))
    plt.close()

    # Plot B: Code Length by Generator (Boxplot is cleaner than hist for many classes)
    plt.figure(figsize=(14, 8))
    # Limiamo gli outlier estremi per rendere il grafico leggibile
    viz_df = df[df['code_length'] < df['code_length'].quantile(0.95)] 
    sns.boxplot(data=viz_df, y='generator', x='code_length', palette="coolwarm", hue='generator', legend=False)
    plt.title(f"Code Length Distribution by Generator (95th percentile) - {dataset_name}")
    plt.xlabel("Character Count")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_PATH, f"{dataset_name}_length_boxplot.png"))
    plt.close()

    # Plot C: Generator vs Language Heatmap
    # Check if some models generate only specific languages
    plt.figure(figsize=(12, 8))
    crosstab = pd.crosstab(df['generator'], df['language'])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap="Blues", cbar=False)
    plt.title(f"Generator vs Language Heatmap - {dataset_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_PATH, f"{dataset_name}_heatmap.png"))
    plt.close()

    print(f"Images saved to {IMG_PATH}")
    
    # ---------------------------------------
    # Helper for Dataset Configuration
    # ---------------------------------------
    if dataset_name == "Train":
        print("\n" + "="*50)
        print("COPY THIS MAP TO src/dataset/dataset.py")
        print("="*50)
        generators = sorted(df['generator'].unique())
        print("GENERATOR_MAP = {")
        for i, gen in enumerate(generators):
            print(f"    '{gen}': {i},")
        print("}")
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

    # Load Test (Sample) if needed
    test_df = load_and_preprocess("test_sample.parquet")
    eda_dataset(test_df, "Test_Sample")