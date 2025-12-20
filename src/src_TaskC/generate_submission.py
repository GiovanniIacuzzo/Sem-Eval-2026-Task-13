import os
import sys
import logging
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from dotenv import load_dotenv
from typing import List, Optional
from copy import deepcopy

# Import PEFT per LoRA
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from src.src_TaskC.models.model import CodeClassifier

# -----------------------------------------------------------------------------
# Configuration & UX
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConsoleUX:
    @staticmethod
    def print_banner(text: str):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

# -----------------------------------------------------------------------------
# Inference Dataset Class
# -----------------------------------------------------------------------------
class InferenceDataset(Dataset):
    """
    Dataset specifico per la generazione della submission.
    Restituisce l'ID originale invece della label.
    """
    def __init__(self, dataframe, tokenizer, max_length, id_col):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id_col = id_col
        
        # Mappa lingue dummy per evitare crash nel forward pass
        self.lang_map_dummy = {'python': 0} 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        code = str(row["code"])
        sample_id = row[self.id_col]
        
        # Semplice troncatura se troppo lungo
        if len(code) > self.max_length * 4:
            code = code[:self.max_length * 4]

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
            "id": sample_id,
            "lang_ids": torch.tensor(0, dtype=torch.long) # Dummy
        }

# -----------------------------------------------------------------------------
# Ensemble Loading Logic (Robust)
# -----------------------------------------------------------------------------
def load_ensemble_models(config, checkpoint_dir, device, k_folds=5) -> List[CodeClassifier]:
    """
    Carica i 5 modelli trained con K-Fold per l'Ensemble.
    """
    models = []
    logger.info(f"Searching for {k_folds} folds in {checkpoint_dir}...")
    
    for fold in range(k_folds):
        fold_dir = os.path.join(checkpoint_dir, f"fold_{fold}")
        if not os.path.exists(fold_dir):
            logger.warning(f"⚠️ Checkpoint for Fold {fold} not found. Skipping.")
            continue
            
        logger.info(f"Loading Fold {fold}...")
        
        # 1. Configurazione Pulita (No auto-LoRA init)
        config_clean = deepcopy(config)
        config_clean["model"]["use_lora"] = False
        
        # 2. Inizializza Modello Base
        model = CodeClassifier(config_clean)
        
        # 3. Applica Pesi (LoRA o Full)
        if config["model"].get("use_lora", False) and PEFT_AVAILABLE:
            # Load Adapter
            model.base_model = PeftModel.from_pretrained(model.base_model, fold_dir)
            
            # Load Head
            head_path = os.path.join(fold_dir, "head.pt")
            if os.path.exists(head_path):
                heads_state = torch.load(head_path, map_location=device)
                model.classifier.load_state_dict(heads_state)
            else:
                logger.error(f"❌ Head missing for Fold {fold}!")
        else:
            # Full Model
            model_path = os.path.join(fold_dir, "model.pt")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            
        model.to(device)
        model.eval()
        models.append(model)
    
    if not models:
        raise ValueError("No models loaded!")
        
    return models

# -----------------------------------------------------------------------------
# Submission Generation Logic
# -----------------------------------------------------------------------------
def run_ensemble_submission(
    models: List[CodeClassifier], 
    test_df: pd.DataFrame, 
    id_col_name: str, 
    output_file: str, 
    device: torch.device,
    max_length: int, 
    batch_size: int = 32
):
    # Usiamo il tokenizer del primo modello
    tokenizer = models[0].tokenizer
    
    dataset = InferenceDataset(
        dataframe=test_df, 
        tokenizer=tokenizer, 
        max_length=max_length, 
        id_col=id_col_name
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    ids: List[str] = []
    final_predictions: List[int] = []

    logger.info(f"Starting Ensemble Inference on {len(dataset)} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # --- FIX: Converti Tensor ID in lista Python ---
            batch_ids = batch["id"]
            if isinstance(batch_ids, torch.Tensor):
                batch_ids = batch_ids.tolist()
            # -----------------------------------------------
            
            # Accumula probabilità da tutti i modelli
            sum_probs = None
            
            for model in models:
                # alpha=0.0 disabilita DANN
                logits, _ = model(input_ids, attention_mask, alpha=0.0)
                probs = torch.softmax(logits, dim=1)
                
                if sum_probs is None:
                    sum_probs = probs
                else:
                    sum_probs += probs
            
            # Media delle probabilità
            avg_probs = sum_probs / len(models)
            
            # Argmax finale
            preds = torch.argmax(avg_probs, dim=1).cpu().tolist()
            
            ids.extend(batch_ids)
            final_predictions.extend(preds)

    # Creazione DataFrame Submission
    submission_df = pd.DataFrame({
        "ID": ids,   # Qui gli ID saranno puliti (es. 437, non tensor(437))
        "label": final_predictions
    })
    
    # Salvataggio
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    submission_df.to_csv(output_file, index=False)
    
    logger.info(f"Submission saved to: {output_file}")
    print(f"\nPreview:\n{submission_df.head().to_string(index=False)}")

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------
def main():
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    ConsoleUX.print_banner("SemEval Task 13 (C) - Submission Generator")

    # 1. Configuration
    config_path = "src/src_TaskC/config/config.yaml"
    checkpoint_dir = "results/results_TaskC/checkpoints"
    
    if not os.path.exists(config_path):
        logger.critical(f"Config file missing: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # 3. Load Ensemble Models
    try:
        models = load_ensemble_models(config, checkpoint_dir, device, k_folds=5)
    except Exception as e:
        logger.critical(f"Model loading failed: {e}")
        sys.exit(1)

    # 4. Data Ingestion
    test_path = "data/Task_C/test.parquet"
    if not os.path.exists(test_path):
        logger.warning(f"Test file {test_path} NOT found. Looking for Kaggle test data...")
        test_path = "data/Task_C/test_sample.parquet" 
        
    if not os.path.exists(test_path):
        logger.error("No test data found! Please put 'test.parquet' in data/Task_C/")
        sys.exit(1)

    logger.info(f"Loading Test Data: {test_path}")
    df = pd.read_parquet(test_path)
    
    # 5. ID Column Detection
    possible_id_cols = ["id", "ID", "sample_id", "row_id"]
    id_col_name = next((col for col in possible_id_cols if col in df.columns), None)
            
    if not id_col_name:
        logger.warning("Missing ID column. Creating fake index IDs (CHECK THIS!).")
        df["id"] = range(len(df))
        id_col_name = "id"
    
    logger.info(f"Using ID Column: '{id_col_name}'")

    # 6. Pre-processing
    df = df.dropna(subset=['code'])
    df['code'] = df['code'].astype(str)

    # 7. Execution
    output_file = "results/results_TaskC/submission/submission_task_c.csv"
    
    max_len = config["data"].get("max_length", 512)
    
    run_ensemble_submission(
        models=models, 
        test_df=df, 
        id_col_name=id_col_name, 
        output_file=output_file, 
        device=device,
        max_length=max_len, 
        batch_size=32 
    )

    ConsoleUX.print_banner("Process Completed Successfully")

if __name__ == "__main__":
    main()