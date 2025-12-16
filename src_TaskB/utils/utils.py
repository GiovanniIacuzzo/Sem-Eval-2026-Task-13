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

    accuracy = accuracy_score(labels, preds)
    
    # Macro F1 is the key metric for SemEval
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='macro', 
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
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
        dtype = torch.bfloat16 # bfloat16 è meglio su CPU moderne, altrimenti float32

    # Disabilita il calcolo dei gradienti per risparmiare memoria
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            # Non-blocking transfer per parallelizzare
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Inference context
            with autocast(device_type=device_type, dtype=dtype):
                logits, loss = model(input_ids, attention_mask, labels=labels)

            if loss is not None:
                running_loss += loss.item()

            # Move to CPU for metrics
            # .detach() è ridondante in no_grad ma buona pratica
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()
            
            predictions.extend(preds)
            references.extend(labels_cpu)
            
            # Non serve cancellare input_ids esplicitamente qui dentro, 
            # Python lo fa a fine scope, ma il gc finale è utile.

    # Calcolo metriche usando il metodo interno del modello (se esiste) o quello generico
    if hasattr(model, "compute_metrics"):
        metrics = model.compute_metrics(predictions, references)
    else:
        metrics = compute_metrics(predictions, references)
        
    metrics["loss"] = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    # PULIZIA CRITICA: Forza il rilascio della memoria GPU
    # Senza questo, potresti avere OOM all'inizio del prossimo epoch di training
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()
    
    return metrics, predictions, references