import os
import logging
import torch
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None

from src.src_TaskB.models.model import CodeClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# PATHS & CONFIG
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Aggiusta questo path se necessario per puntare alla root del progetto
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

BINARY_CKPT = os.path.join(PROJECT_ROOT, "results/results_TaskB/checkpoints/binary")
FAMILIES_CKPT = os.path.join(PROJECT_ROOT, "results/results_TaskB/checkpoints/families")
TEST_PATH = os.path.join(PROJECT_ROOT, "data/Task_B/test.parquet")
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "results/results_TaskB/submission")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# IMPORTANTE: num_extra_features DEVE ESSERE 8 (come nel training)
BINARY_CFG = {
    "model": {
        "model_name": "microsoft/unixcoder-base", 
        "num_labels": 2, 
        "use_lora": False,
        "num_extra_features": 8 
    }
}
FAMILIES_CFG = {
    "model": {
        "model_name": "microsoft/unixcoder-base", 
        # Modificato a 10 perché il tuo checkpoint ha shape [10, 768]
        "num_labels": 10, 
        "use_lora": False, # Metti True se il checkpoint famiglie usa LoRA, altrimenti False
        "lora_r": 64,
        "num_extra_features": 8 
    }
}

# -----------------------------------------------------------------------------
# STYLOMETRY & DATASET (Identico al Training)
# -----------------------------------------------------------------------------
def extract_stylometric_features(code):
    """
    Deve corrispondere esattamente alla logica usata durante il training.
    Restituisce un vettore di 8 dimensioni.
    """
    features = []
    lines = code.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    code_len = len(code) + 1
    
    # 1. Ratio Spazi
    features.append(code.count(' ') / code_len)
    
    # 2. Ratio Commenti
    features.append((code.count('#') + code.count('//')) / code_len)
    
    # 3. Ratio Simboli Speciali
    features.append(len(re.findall(r'[{}()\[\];.,]', code)) / code_len)
    
    # 4. Lunghezza Media Righe
    avg_line_len = np.mean([len(l) for l in non_empty_lines]) if non_empty_lines else 0
    features.append(min(avg_line_len / 100.0, 1.0))
    
    # 5. Ratio Righe Vuote
    features.append((len(lines) - len(non_empty_lines)) / (len(lines) + 1))
    
    # 6. Snake vs Camel Case
    snake_count = code.count('_')
    camel_count = len(re.findall(r'[a-z][A-Z]', code))
    features.append(snake_count / (snake_count + camel_count + 1))
    
    # 7. Token Logici
    logic_tokens = len(re.findall(r'\b(if|for|while|return|switch|case|break)\b', code))
    features.append(logic_tokens / (len(code.split()) + 1))
    
    # 8. Indentazione Massima
    max_indent = max([len(l) - len(l.lstrip()) for l in non_empty_lines]) if non_empty_lines else 0
    features.append(min(max_indent / 20.0, 1.0))
    
    return features

class SubmissionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info("Extracting stylometric features (8 dimensions)...")
        # Estrazione features
        raw_features = [extract_stylometric_features(str(c)) for c in tqdm(self.data['code'], desc="Features")]

        scaler = StandardScaler()
        self.features = scaler.fit_transform(raw_features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = str(self.data.at[idx, 'code'])
        extra_feat = self.features[idx] # Array di float di dim 8

        encoding = self.tokenizer(
            code, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "extra_features": torch.tensor(extra_feat, dtype=torch.float)
        }

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
def load_model(checkpoint_dir, config, device, name):
    logger.info(f"[{name}] Loading model logic...")
    
    # Inizializza modello pulito
    model = CodeClassifier(config)
    model.to(device)
    
    # Percorsi possibili per i pesi
    # 1. classifier.pt (caso PEFT/Custom save)
    # 2. full_model.bin (caso Standard save)
    cls_path = os.path.join(checkpoint_dir, "classifier.pt")
    full_path = os.path.join(checkpoint_dir, "full_model.bin")
    
    loaded = False
    
    # Tentativo caricamento Full Model
    if os.path.exists(full_path):
        logger.info(f"[{name}] Found full_model.bin, loading...")
        state_dict = torch.load(full_path, map_location=device)
        
        # Pulizia chiavi 'module.' se presenti
        new_state_dict = {}
        for k, v in state_dict.items():
            name_key = k.replace("module.", "")
            new_state_dict[name_key] = v
            
        try:
            model.load_state_dict(new_state_dict, strict=False)
            loaded = True
        except Exception as e:
            logger.error(f"[{name}] Error loading full model: {e}")

    # Tentativo caricamento PEFT/Classifier solo
    elif os.path.exists(cls_path):
        logger.info(f"[{name}] Found classifier.pt, loading head...")
        state_dict = torch.load(cls_path, map_location=device)
        model.classifier.load_state_dict(state_dict)
        # Se usavi LoRA, qui dovresti caricare anche l'adapter, ma il tuo codice base 
        # CodeClassifier ricarica il base_model pulito. Se hai l'adapter salvato:
        if config["model"].get("use_lora"):
             try:
                 model.base_model.load_adapter(checkpoint_dir)
                 logger.info(f"[{name}] LoRA adapters loaded.")
             except:
                 logger.warning(f"[{name}] No LoRA adapters found/loaded.")
        loaded = True

    if not loaded:
        logger.error(f"[{name}] CRITICAL: No weights found in {checkpoint_dir} (checked full_model.bin and classifier.pt)")
        # Non crashiamo qui per debug, ma il risultato sarà random
        
    model.eval()
    return model

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Generating Submission using device: {device}")
    
    if not os.path.exists(TEST_PATH):
        logger.error(f"Test file not found at {TEST_PATH}")
        return

    df = pd.read_parquet(TEST_PATH)
    # Se il dataframe è enorme, puoi ridurlo per debug: df = df.head(100)
    df = df.reset_index(drop=True)
    logger.info(f"Loaded {len(df)} test samples.")
    
    # Identify ID Column
    possible_ids = ['id', 'ID', 'sample_id']
    id_col = next((col for col in possible_ids if col in df.columns), None)
    if not id_col:
        logger.warning("ID column missing. Generating sequential IDs.")
        df['id'] = df.index
        id_col = 'id'

    # --- FASE 1: Binary Classification ---
    logger.info(">>> STEP 1: Binary Classification (Human vs AI)")
    tok_bin = AutoTokenizer.from_pretrained(BINARY_CFG["model"]["model_name"])
    
    # Creiamo dataset
    ds_bin = SubmissionDataset(df, tok_bin)
    dl_bin = DataLoader(ds_bin, batch_size=32, shuffle=False, num_workers=4)
    
    model_bin = load_model(BINARY_CKPT, BINARY_CFG, device, "Binary")
    
    is_ai_probs = []
    with torch.no_grad():
        for batch in tqdm(dl_bin, desc="Binary Infer"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            extra = batch["extra_features"].to(device)
            
            # Forward: logits, loss, features
            logits, _, _ = model_bin(input_ids, mask, extra_features=extra)
            probs = torch.softmax(logits, dim=1)
            # Binary: index 1 è AI
            is_ai_probs.extend(probs[:, 1].cpu().tolist())
    
    del model_bin
    torch.cuda.empty_cache()

    # Threshold 0.5
    is_ai_preds = [1 if p > 0.5 else 0 for p in is_ai_probs]
    num_ai = sum(is_ai_preds)
    logger.info(f"Split Result -> Human: {len(df) - num_ai} | AI: {num_ai}")

    # --- FASE 2: Families Classification ---
    logger.info(">>> STEP 2: Families Classification (AI Attribution)")
    final_preds_map = {} # Mappa: index_originale -> predizione_classe
    
    # Filtriamo solo gli indici predetti come AI
    ai_indices = [i for i, x in enumerate(is_ai_preds) if x == 1]
    
    if len(ai_indices) > 0:
        # Creiamo un sotto-dataframe solo con le righe AI
        df_ai = df.iloc[ai_indices].reset_index(drop=True)
        # Mappiamo l'indice del nuovo df_ai all'indice originale di df
        original_idx_map = df.iloc[ai_indices].index.tolist()
        
        tok_fam = AutoTokenizer.from_pretrained(FAMILIES_CFG["model"]["model_name"])
        ds_fam = SubmissionDataset(df_ai, tok_fam, max_length=512)
        dl_fam = DataLoader(ds_fam, batch_size=32, shuffle=False, num_workers=4)
        
        model_fam = load_model(FAMILIES_CKPT, FAMILIES_CFG, device, "Families")
        
        local_ptr = 0
        with torch.no_grad():
            for batch in tqdm(dl_fam, desc="Families Infer"):
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                extra = batch["extra_features"].to(device)
                
                logits, _, _ = model_fam(input_ids, mask, extra_features=extra)
                # Argmax per ottenere la classe 0..9
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                
                for p in preds:
                    orig_idx = original_idx_map[local_ptr]
                    # Logica Mapping:
                    # Il modello outputta 0..9. Noi mappiamo a 1..10.
                    # 0 -> 1, 1 -> 2, ecc.
                    final_preds_map[orig_idx] = p + 1
                    local_ptr += 1
        
        del model_fam
        torch.cuda.empty_cache()

    # --- FASE 3: Output ---
    logger.info(">>> STEP 3: Saving Submission")
    final_labels = []
    
    for i in range(len(df)):
        if is_ai_preds[i] == 0:
            final_labels.append(0) # Human
        else:
            # Se era AI, prendiamo la label calcolata, altrimenti default 1
            final_labels.append(final_preds_map.get(i, 1))

    submission = pd.DataFrame({
        "ID": df[id_col],
        "label": final_labels
    })
    
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(SUBMISSION_FILE, index=False)
    
    logger.info(f"Submission saved successfully to: {SUBMISSION_FILE}")
    print("\nFinal Label Distribution:")
    print(submission['label'].value_counts().sort_index())

if __name__ == "__main__":
    main()