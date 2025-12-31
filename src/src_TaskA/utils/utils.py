import torch
import numpy as np
import gc
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, classification_report
from torch.amp import autocast
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Metric Computation
# -----------------------------------------------------------------------------
def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    preds = np.array(preds)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, preds)
    
    precision_m, recall_m, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    
    precision_b, recall_b, f1_binary, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "f1": float(f1_macro),
        "f1_binary": float(f1_binary),
        "precision": float(precision_m),
        "recall": float(recall_m)
    }

# -----------------------------------------------------------------------------
# Evaluation Loop
# -----------------------------------------------------------------------------
def evaluate(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False, dynamic_ncols=True):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                # Il modello HF ritorna (loss, logits) se labels sono passate,
                # ma per sicurezza ricalcoliamo o usiamo l'output standard
                outputs = model(input_ids, attention_mask)
                # SimpleCodeClassifier ritorna (loss, logits) se labels sono passate al forward,
                # oppure solo logits se no. Adattiamo in base all'implementazione model.py
                
                # Assumendo model.py ritorni (loss, logits)
                logits = outputs[1] 
                loss = criterion(logits, labels)
            
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = val_loss / len(dataloader)
    
    # Metriche
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary') # Task A è binario
    
    # Report dettagliato per log
    report = classification_report(all_labels, all_preds, target_names=["Human", "Machine"], output_dict=True)
    
    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "f1": f1,
        "human_f1": report["Human"]["f1-score"],
        "machine_f1": report["Machine"]["f1-score"]
    }
    
    return metrics
