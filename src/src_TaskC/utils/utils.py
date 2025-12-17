import torch
import numpy as np
import gc
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import autocast
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Metric Computation (ADATTATO PER TASK C)
# -----------------------------------------------------------------------------
def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Computes classification metrics for Multiclass Task (Human, AI, Hybrid, Adv).
    Primary Metric: Macro F1-score.
    """
    preds = np.array(preds)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, preds)
    
    # Macro F1 è la metrica ufficiale per il Task C
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),         
        "precision": float(precision),
        "recall": float(recall)
    }

# -----------------------------------------------------------------------------
# Evaluation Loop (Nessun cambiamento necessario, mantiene alpha=0.0)
# -----------------------------------------------------------------------------
def evaluate(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device
) -> Tuple[Dict[str, float], List[int], List[int]]:
    """
    Runs evaluation loop. Sets alpha=0 to disable DANN during validation.
    """
    model.eval()
    running_loss = 0.0
    predictions = []
    references = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False, dynamic_ncols=True)

    if device.type == 'cuda':
        device_type = 'cuda'
        dtype = torch.float16
    elif device.type == 'mps':
        device_type = 'mps'
        dtype = torch.float16
    else:
        device_type = 'cpu'
        dtype = torch.bfloat16

    with torch.no_grad():
        for batch in progress_bar:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)
            
            with autocast(device_type=device_type, dtype=dtype):
                # IMPORTANTE: alpha=0.0 disabilita il ramo avversario (DANN).
                logits, loss = model.forward(
                    input_ids, 
                    attention_mask, 
                    labels=labels, 
                    alpha=0.0 
                )
            
            if loss is not None:
                running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.detach().cpu().numpy())
            references.extend(labels.detach().cpu().numpy())
            
            del input_ids, attention_mask, labels, logits, loss

    eval_metrics = compute_metrics(predictions, references)
    eval_metrics["loss"] = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    return eval_metrics, predictions, references