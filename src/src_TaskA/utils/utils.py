import torch
import numpy as np
import gc
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import autocast
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Metric Computation
# -----------------------------------------------------------------------------
def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Computes classification metrics for Binary Task (Human vs AI).
    """
    preds = np.array(preds)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, preds)
    
    # Task A is Binary. 
    # 'binary' calcola F1 solo per la classe positiva (AI).
    # 'macro' è la media non pesata tra Human e AI.
    # SemEval spesso guarda la Macro F1 anche per il binario per penalizzare il bias.
    precision_m, recall_m, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    
    # Calcoliamo anche la Binary per riferimento interno (performance sull'AI)
    precision_b, recall_b, f1_binary, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "f1": float(f1_macro),         # Metrica Ufficiale (Macro)
        "f1_binary": float(f1_binary), # Metrica Utile (AI detection)
        "precision": float(precision_m),
        "recall": float(recall_m)
    }

# -----------------------------------------------------------------------------
# Evaluation Loop (T4 Optimized)
# -----------------------------------------------------------------------------
def evaluate(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device
) -> Tuple[Dict[str, float], List[int], List[int]]:
    """
    Runs evaluation loop.
    Crucial: Sets alpha=0 to disable Adversarial/DANN during validation.
    """
    model.eval()
    running_loss = 0.0
    predictions = []
    references = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False, dynamic_ncols=True)

    # Rilevamento device robusto
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
            # Non-blocking transfer
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)
            # Lang IDs non servono per la metrica di classificazione, 
            # ma se volessimo calcolare la DANN loss in validation potremmo passarli.
            # Per ora li ignoriamo per risparmiare banda.
            
            # Mixed Precision Context
            with autocast(device_type=device_type, dtype=dtype):
                # IMPORTANTE: alpha=0.0 disabilita il ramo avversario.
                # In validation vogliamo solo sapere se il modello distingue Human vs AI.
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
            
            # Non serve cancellare manualmente qui, ma aiuta il garbage collector
            del input_ids, attention_mask, labels, logits, loss

    # Compute metrics
    eval_metrics = compute_metrics(predictions, references)
    eval_metrics["loss"] = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    # PULIZIA MEMORIA CRITICA PER T4
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    return eval_metrics, predictions, references