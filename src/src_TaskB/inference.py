import os
import sys
import logging
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from torch.amp import autocast
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any

# Local imports
from src_TaskB.models.model import CodeClassifier
from src_TaskB.dataset.dataset import CodeDataset, load_base_dataframe, GENERATOR_MAP
from src_TaskB.utils.utils import compute_metrics

# PEFT import
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Inference Logic
# -----------------------------------------------------------------------------
def run_inference(model, test_df, device, max_length, batch_size=32):
    """
    Esegue l'inferenza sul dataframe fornito.
    Richiede max_length esplicito dalla config.
    """
    model.eval()
    
    # Dataset in modalità 'val' (o 'test' se cieco, ma qui assumiamo validation logic)
    dataset = CodeDataset(
        dataframe=test_df,
        tokenizer=model.tokenizer,
        max_length=max_length,  # <--- FIX: Usiamo il valore passato, non model.max_length
        mode="val" 
    )
    
    # Num workers basso per evitare overhead su dataset piccoli
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    all_preds, all_refs = [], []
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    dtype = torch.float16 if device_type == 'cuda' else torch.float32

    logger.info(f"Starting inference on {len(dataset)} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Gestione input DANN (lang_ids)
            lang_ids = batch.get("lang_ids", None)
            if lang_ids is not None:
                lang_ids = lang_ids.to(device)

            with autocast(device_type=device_type, dtype=dtype):
                # alpha=0.0 disabilita il ramo avversario in inferenza
                logits, _ = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    lang_ids=lang_ids, 
                    alpha=0.0 
                )

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_refs.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_preds, all_refs)
    return all_preds, all_refs, metrics

# -----------------------------------------------------------------------------
# Reporting Functions
# -----------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Ricostruiamo la mappa inversa
    inv_map = {v: k for k, v in GENERATOR_MAP.items()}
    
    # Creiamo le label ordinate per l'asse (0, 1, 2, ... 30)
    sorted_indices = sorted(GENERATOR_MAP.values())
    tick_labels = [inv_map[i] for i in sorted_indices]
    
    # Calcolo CM
    cm = confusion_matrix(y_true, y_pred, labels=sorted_indices)
    
    plt.figure(figsize=(24, 20)) # Aumentato per 31 classi
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title("Confusion Matrix - Task B")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90, fontsize=8) 
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion Matrix saved to {save_path}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    
    # 1. Configurazione
    config_path = "src_TaskB/config/config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Inizializzazione Architettura
    model_wrapper = CodeClassifier(config)
    model_wrapper.to(device)

    # 3. Caricamento Pesi
    checkpoint_dir = os.path.abspath("results_TaskB/checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    is_peft = config["model"].get("use_lora", True)

    if is_peft:
        logger.info(f"Loading LoRA Adapters from: {checkpoint_dir}")
        
        # A. Carica Adapter LoRA
        model_wrapper.base_model.load_adapter(checkpoint_dir, adapter_name="default")
        model_wrapper.base_model.set_adapter("default")
        
        # B. Carica Componenti Custom
        custom_path = os.path.join(checkpoint_dir, "custom_components.pt")
        logger.info(f"Loading Custom Heads from: {custom_path}")
        
        custom_state = torch.load(custom_path, map_location=device)
        
        # Caricamento sicuro delle componenti
        model_wrapper.classifier.load_state_dict(custom_state['classifier'])
        
        if 'pooler' in custom_state:
            model_wrapper.pooler.load_state_dict(custom_state['pooler'])
        else:
            logger.warning("'pooler' key not found in checkpoint.")

        if 'projection_head' in custom_state:
            model_wrapper.projection_head.load_state_dict(custom_state['projection_head'])
        
        if 'language_classifier' in custom_state:
            model_wrapper.language_classifier.load_state_dict(custom_state['language_classifier'])

        logger.info("LoRA + Custom Heads loaded successfully.")
    else:
        full_path = os.path.join(checkpoint_dir, "full_model.bin")
        model_wrapper.load_state_dict(torch.load(full_path, map_location=device))

    # 4. Caricamento Dati
    test_path = "data/Task_B/test_sample.parquet"
    if not os.path.exists(test_path):
        logger.error(f"Test file not found: {test_path}")
        sys.exit(1)
        
    logger.info(f"Loading test data from {test_path}...")
    test_df = load_base_dataframe(test_path, task_type="multiclass")
    
    # 5. Esecuzione
    # FIX: Passiamo max_length dalla config
    max_len = config["data"]["max_length"]
    preds, refs, metrics = run_inference(model_wrapper, test_df, device, max_length=max_len)
    
    # 6. Report Finale
    print("\n" + "="*60)
    print("FINAL EVALUATION REPORT")
    print("="*60)
    
    all_class_ids = sorted(GENERATOR_MAP.values())
    inv_map = {v: k for k, v in GENERATOR_MAP.items()}
    target_names = [inv_map[i] for i in all_class_ids]
    
    print(classification_report(
        refs, 
        preds, 
        labels=all_class_ids, 
        target_names=target_names, 
        zero_division=0
    ))
    
    output_dir = "results_TaskB/inference_output"
    plot_confusion_matrix(refs, preds, os.path.join(output_dir, "confusion_matrix.png"))
    
    results_df = pd.DataFrame({
        "True Label": [inv_map.get(r, "Unknown") for r in refs],
        "Predicted Label": [inv_map.get(p, "Unknown") for p in preds],
        "Correct": [r == p for r, p in zip(refs, preds)]
    })
    results_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    logger.info(f"Detailed predictions saved to {output_dir}/predictions.csv")

    logger.info("Inference completed.")