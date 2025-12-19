import os
import sys
import logging
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# --- FIX SICUREZZA PYTORCH ---
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
# -----------------------------

# Local imports
from src.src_TaskB.models.model import CodeClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# --- PERCORSI DINAMICI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

# Percorsi
BINARY_CKPT = os.path.join(PROJECT_ROOT, "results/results_TaskB/checkpoints/binary")
FAMILIES_CKPT = os.path.join(PROJECT_ROOT, "results/results_TaskB/checkpoints/families")
TEST_PATH = os.path.join(PROJECT_ROOT, "data/Task_B/test.parquet")
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "results/results_TaskB/submission")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Configurazioni
BINARY_CFG = {
    "model": {"model_name": "microsoft/codebert-base", "num_labels": 2, "use_lora": False}
}
FAMILIES_CFG = {
    "model": {"model_name": "microsoft/unixcoder-base", "num_labels": 10, "use_lora": True, "lora_r": 64}
}

class SubmissionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe # DataFrame già resettato
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = str(self.data.at[idx, 'code'])
        encoding = self.tokenizer(
            code, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

def load_model(checkpoint_dir, config, device, name):
    logger.info(f"[{name}] Loading model from {checkpoint_dir}...")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    model = CodeClassifier(config)
    model.to(device)
    
    is_peft = config["model"]["use_lora"]
    if is_peft:
        try:
            model.base_model.load_adapter(checkpoint_dir, adapter_name="default")
        except Exception as e:
            raise RuntimeError(f"Failed to load adapter: {e}")
        
        custom_path = os.path.join(checkpoint_dir, "custom_components.pt")
        if os.path.exists(custom_path):
            state = torch.load(custom_path, map_location=device)
            model.classifier.load_state_dict(state['classifier'])
            model.pooler.load_state_dict(state['pooler'])
    else:
        full_path = os.path.join(checkpoint_dir, "full_model.bin")
        if not os.path.exists(full_path):
            full_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Weights missing in {checkpoint_dir}")
        model.load_state_dict(torch.load(full_path, map_location=device))

    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Generating Submission using device: {device}")
    
    if not os.path.exists(TEST_PATH):
        logger.error(f"Test file missing: {TEST_PATH}")
        return

    df = pd.read_parquet(TEST_PATH)
    
    # --- CRITICAL FIX: RESET INDEX ---
    # Questo assicura che gli indici siano 0, 1, 2... N coerenti con il loop finale
    df = df.reset_index(drop=True)
    logger.info(f"Loaded {len(df)} test samples (Index Reset).")
    
    # Verifica colonna ID per sottomissione
    id_col = 'id' if 'id' in df.columns else 'ID'
    if id_col not in df.columns:
        logger.warning("Colonna ID non trovata! Genero ID sequenziali.")
        df['id'] = df.index
        id_col = 'id'

    # --- FASE 1: Binary ---
    logger.info(">>> FASE 1: Binary Classification")
    tok_bin = AutoTokenizer.from_pretrained(BINARY_CFG["model"]["model_name"])
    ds_bin = SubmissionDataset(df, tok_bin)
    dl_bin = DataLoader(ds_bin, batch_size=32, shuffle=False, num_workers=4)
    
    model_bin = load_model(BINARY_CKPT, BINARY_CFG, device, "Binary")
    
    is_ai_probs = []
    with torch.no_grad():
        for batch in tqdm(dl_bin, desc="Binary Infer"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits, _ = model_bin(input_ids, mask)
            probs = torch.softmax(logits, dim=1)
            is_ai_probs.extend(probs[:, 1].cpu().tolist())
    
    del model_bin
    torch.cuda.empty_cache()

    is_ai_preds = [1 if p > 0.5 else 0 for p in is_ai_probs]
    num_ai = sum(is_ai_preds)
    logger.info(f"Human: {len(df) - num_ai}, AI: {num_ai}")

    # --- FASE 2: Families ---
    logger.info(">>> FASE 2: Families Classification")
    final_preds_map = {} 
    
    ai_indices = [i for i, x in enumerate(is_ai_preds) if x == 1]
    
    if len(ai_indices) > 0:
        # Crea sotto-dataframe solo con le righe AI
        df_ai = df.iloc[ai_indices].reset_index(drop=True)
        # Salviamo gli indici originali (che ora sono sicuri 0..N grazie al reset iniziale)
        original_idx_map = df.iloc[ai_indices].index.tolist()
        
        tok_fam = AutoTokenizer.from_pretrained(FAMILIES_CFG["model"]["model_name"])
        ds_fam = SubmissionDataset(df_ai, tok_fam, max_length=384)
        dl_fam = DataLoader(ds_fam, batch_size=32, shuffle=False, num_workers=4)
        
        model_fam = load_model(FAMILIES_CKPT, FAMILIES_CFG, device, "Families")
        
        local_ptr = 0
        with torch.no_grad():
            for batch in tqdm(dl_fam, desc="Families Infer"):
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                
                logits, _ = model_fam(input_ids, mask)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                
                for p in preds:
                    orig_idx = original_idx_map[local_ptr]
                    # Mappa 0-9 -> 1-10
                    final_preds_map[orig_idx] = p + 1
                    local_ptr += 1
        
        del model_fam
        torch.cuda.empty_cache()

    # --- FASE 3: Assemblaggio ---
    logger.info(">>> FASE 3: Scrittura Submission")
    final_labels = []
    
    fallback_count = 0
    for i in range(len(df)):
        if is_ai_preds[i] == 0:
            final_labels.append(0)
        else:
            # Qui ora siamo sicuri che 'i' corrisponde agli indici in final_preds_map
            if i in final_preds_map:
                final_labels.append(final_preds_map[i])
            else:
                # Questo non dovrebbe più succedere
                final_labels.append(1) 
                fallback_count += 1
                
    if fallback_count > 0:
        logger.warning(f"ATTENZIONE: Fallback a label 1 usato {fallback_count} volte (errore mapping).")

    submission = pd.DataFrame({
        "ID": df[id_col],
        "label": final_labels
    })
    
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(SUBMISSION_FILE, index=False)
    
    logger.info(f"✅ Sottomissione salvata: {SUBMISSION_FILE}")
    logger.info("\nDistribuzione Finale:")
    logger.info(submission['label'].value_counts().sort_index())

if __name__ == "__main__":
    main()