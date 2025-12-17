import torch
import numpy as np
import gc
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import autocast
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Metric Computation Strategy
# -----------------------------------------------------------------------------
def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Computes classification metrics aligned with SemEval competition standards.
    """
    preds = np.array(preds)
    labels = np.array(labels)

    # Calcolo base
    accuracy = accuracy_score(labels, preds)
    
    # Macro F1: La metrica Ufficiale della competizione (tratta tutte le classi uguali)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='macro', 
        zero_division=0
    )

    # Weighted F1: La metrica "Reale" (tiene conto di quanti esempi ci sono per classe)
    # Se Weighted è alto (0.90) ma Macro è basso (0.30), il modello predice solo "Human"!
    # Se entrambi salgono, il modello sta imparando davvero.
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='weighted', 
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1": float(f1_macro),          # Target SemEval
        "f1_weighted": float(f1_weighted) # Per debug sbilanciamento
    }

# -----------------------------------------------------------------------------
# Inference & Evaluation Loop (Optimized for T4/CUDA)
# -----------------------------------------------------------------------------
def evaluate(model, dataloader, device) -> Tuple[Dict[str, float], List[int], List[int]]:
    """
    Validation loop optimized for inference stability on CUDA.
    """
    model.eval()
    predictions, references = [], []
    running_loss = 0.0
    
    # Rilevamento device type robusto per Autocast
    if device.type == 'cuda':
        device_type = 'cuda'
        dtype = torch.float16
    elif device.type == 'mps':
        device_type = 'mps'
        dtype = torch.float16
    else:
        device_type = 'cpu'
        dtype = torch.bfloat16 

    # Disabilita il calcolo dei gradienti
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            # Spostamento su GPU
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # Recuperiamo anche lang_ids se presente (per il DANN)
            # Default a None se non c'è nel batch
            lang_ids = batch.get("lang_ids", None)
            if lang_ids is not None:
                lang_ids = lang_ids.to(device, non_blocking=True)

            # Inference context
            with autocast(device_type=device_type, dtype=dtype):
                # Passiamo alpha=0.0 perché in validazione NON vogliamo 
                # l'inversione del gradiente (adversarial), vogliamo solo valutare.
                logits, loss = model(
                    input_ids, 
                    attention_mask, 
                    lang_ids=lang_ids, 
                    labels=labels, 
                    alpha=0.0 
                )

            if loss is not None:
                running_loss += loss.item()

            # Move to CPU for metrics
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()
            
            predictions.extend(preds)
            references.extend(labels_cpu)

    # Calcolo metriche
    if hasattr(model, "compute_metrics"):
        # Usa quella interna al modello se c'è
        metrics = model.compute_metrics(predictions, references)
    else:
        metrics = compute_metrics(predictions, references)
        
    metrics["loss"] = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    # Pulizia memoria
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()
    
    return metrics, predictions, references