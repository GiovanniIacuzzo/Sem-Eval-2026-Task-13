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
def evaluate(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device
) -> Tuple[Dict[str, float], List[int], List[int]]:
    """
    Esegue il loop di valutazione passando anche le feature stilometriche.
    """
    model.eval()
    running_loss = 0.0
    predictions = []
    references = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False, dynamic_ncols=True)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)
            
            stylo_feats    = batch.get("stylo_feats")
            if stylo_feats is not None:
                stylo_feats = stylo_feats.to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                logits, loss = model(
                    input_ids, 
                    attention_mask, 
                    stylo_feats=stylo_feats,
                    labels=labels
                )
            
            if loss is not None:
                running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.detach().cpu().numpy())
            references.extend(labels.detach().cpu().numpy())
            
            # Pulizia per evitare picchi di VRAM
            del input_ids, attention_mask, labels, logits, loss, stylo_feats

    eval_metrics = compute_metrics(predictions, references)
    eval_metrics["loss"] = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    # Pulizia memoria GPU aggressiva post-valutazione
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    gc.collect()

    return eval_metrics, predictions, references