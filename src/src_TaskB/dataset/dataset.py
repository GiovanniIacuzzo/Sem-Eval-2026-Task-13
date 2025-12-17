import random
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. FAMILY MAP (Il cuore della correzione)
# -----------------------------------------------------------------------------
# Mappiamo i generatori specifici alle 11 Famiglie richieste dal Task.
# 0: Human
# 1-10: AI Families
# -----------------------------------------------------------------------------
FAMILY_MAP = {
    # --- Class 0: HUMAN ---
    'human': 0,

    # --- Class 1: 01-AI ---
    '01-ai/yi-coder-1.5b': 1, 
    '01-ai/yi-coder-1.5b-chat': 1,

    # --- Class 2: BigCode ---
    'bigcode/starcoder': 2, 
    'bigcode/starcoder2-15b': 2, 
    'bigcode/starcoder2-3b': 2,

    # --- Class 3: DeepSeek-AI ---
    'deepseek-ai/deepseek-coder-1.3b-base': 3, 
    'deepseek-ai/deepseek-r1': 3, 
    'deepseek-ai/deepseek-v3-0324': 3,

    # --- Class 4: Gemma (Google) ---
    'gemma-3-27b-it': 4, 
    'gemma-3n-e4b-it': 4, 
    'google/codegemma-2b': 4,

    # --- Class 5: Phi (Microsoft) ---
    'microsoft/phi-3-medium-4k-instruct': 5, 
    'microsoft/phi-3-mini-4k-instruct': 5, 
    'microsoft/phi-3-small-8k-instruct': 5,

    # --- Class 6: Meta-LLaMA ---
    'meta-llama/llama-3.1-8b': 6, 
    'meta-llama/llama-3.1-8b-instruct': 6, 
    'meta-llama/llama-3.2-11b-vision-instruct': 6,
    'meta-llama/llama-3.2-1b': 6, 
    'meta-llama/llama-3.2-3b': 6,

    # --- Class 7: IBM-Granite ---
    'ibm-granite/granite-3.2-2b-instruct': 7, 
    'ibm-granite/granite-3.3-8b-base': 7, 
    'ibm-granite/granite-3.3-8b-instruct': 7,

    # --- Class 8: Mistral ---
    'mistralai/devstral-small-2505': 8, 
    'mistralai/mistral-7b-instruct-v0.3': 8,

    # --- Class 9: Qwen ---
    'qwen/qwen2.5-72b-instruct': 9, 
    'qwen/qwen2.5-codder-14b-instruct': 9, 
    'qwen/qwen2.5-coder-1.5b': 9,
    'qwen/qwen2.5-coder-1.5b-instruct': 9, 
    'qwen/qwq-32b': 9,

    # --- Class 10: OpenAI ---
    'gpt-4o': 10
}

# -----------------------------------------------------------------------------
# Data Loading Helper
# -----------------------------------------------------------------------------
def load_base_dataframe(file_path: str):
    """
    Carica il parquet e applica la mappatura per Famiglie.
    """
    if not file_path.endswith('.parquet'):
        raise ValueError("Only .parquet files are supported")
        
    df = pd.read_parquet(file_path)
    
    # Normalizzazione stringhe
    if 'generator' in df.columns:
        df['generator'] = df['generator'].astype(str).str.lower().str.strip()
        df['generator'] = df['generator'].str.replace(r'\s+', ' ', regex=True)

    if 'language' in df.columns:
        df['language'] = df['language'].astype(str).str.lower().str.strip()

    # Applicazione Mappa Famiglie
    if 'generator' in df.columns:
        # Usa la FAMILY_MAP definita sopra
        df['label'] = df['generator'].map(FAMILY_MAP)
        
        # Gestione Unknowns (nuovi modelli non mappati esplicitamente ma simili)
        # Se nel training trovi NaN, è un problema. Nel test, li ignoreremo o li gestiamo dopo.
        unknowns_count = df['label'].isna().sum()
        if unknowns_count > 0:
            logger.warning(f"⚠️ {unknowns_count} samples could not be mapped to a known family!")
            # Per sicurezza li mettiamo a -1
            df['label'] = df['label'].fillna(-1)
            
        df['label'] = df['label'].astype(int)
    else:
        df['label'] = -1

    return df

# -----------------------------------------------------------------------------
# Dataset Class (Invariata ma robusta)
# -----------------------------------------------------------------------------
class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, language_map=None, max_length=512, augment=False, mode="train"):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Gestione augment
        if mode == "train":
            self.augment = True
        elif mode in ["val", "test"]:
            self.augment = False
        else:
            self.augment = augment

        self.language_map = language_map if language_map else {}

    def __len__(self):
        return len(self.data)

    def _structural_noise(self, code):
        # Noise leggero per evitare overfitting sui 442k human samples
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            r = random.random()
            if r < 0.05: continue # Drop line raro
            if r < 0.10: new_lines.append("") # Insert empty line
            if r < 0.02: line = "# " + line # Comment out (Python style, crude but effective noise)
            new_lines.append(line)
        return "\n".join(new_lines)

    def __getitem__(self, idx):
        code = str(self.data.at[idx, 'code'])
        
        # Label (0-10) o -1
        gen_label = int(self.data.at[idx, 'label']) if 'label' in self.data.columns else -1
        
        # Lang ID per DANN
        lang_str = str(self.data.at[idx, 'language']).lower() if 'language' in self.data.columns else ""
        lang_id = self.language_map.get(lang_str, -1)

        # Augmentation (Solo Training)
        if self.augment:
            if random.random() < 0.3: # Applica solo al 30% dei campioni
                code = self._structural_noise(code)
            
            # Random Crop se troppo lungo
            if len(code) > self.max_length * 4:
                start = random.randint(0, len(code) - int(self.max_length * 3.5))
                code = code[start : start + int(self.max_length * 4)]

        encoding = self.tokenizer(
            code, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(gen_label, dtype=torch.long),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long)
        }

# -----------------------------------------------------------------------------
# Main Loader with Class Weighting
# -----------------------------------------------------------------------------
def load_data(config, tokenizer):
    """
    Carica i dati, calcola i pesi per le classi sbilanciate e restituisce tutto.
    """
    df = load_base_dataframe(config["data"]["train_path"])
    
    # 1. Pulizia base
    df = df.dropna(subset=['code'])
    df = df[df['code'].str.len() > 10] # Codice troppo corto è rumore
    df = df[df['label'] != -1] # Rimuovi classi non mappate dal training
    
    # 2. Setup Mappa Linguaggi (per DANN)
    target_langs = config["model"].get("languages", [])
    language_map = {l: i for i, l in enumerate(target_langs)}
    
    # 3. Calcolo Pesi per Bilanciamento (CRUCIALE)
    # Human (0) ha 442k sample, le altre poche migliaia.
    # Senza pesi, il modello predirà sempre 0.
    labels = df['label'].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    # Converti in tensore float per PyTorch
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    logger.info(f"Class Weights Computed: {class_weights_tensor}")

    # 4. Split Stratificato
    # Assicura che ogni batch di validazione abbia tutte le 11 famiglie
    train_df, val_df = train_test_split(
        df, test_size=0.10, stratify=df['label'], random_state=42
    )

    logger.info(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    train_ds = CodeDataset(train_df, tokenizer, language_map, config["data"]["max_length"], mode="train")
    val_ds = CodeDataset(val_df, tokenizer, language_map, config["data"]["max_length"], mode="val")
    
    # Restituiamo anche i pesi!
    return train_ds, val_ds, class_weights_tensor