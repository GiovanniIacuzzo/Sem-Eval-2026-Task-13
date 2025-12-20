import torch
import numpy as np
import gc
import logging
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.amp import autocast
from tqdm import tqdm

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Metric Computation
# -----------------------------------------------------------------------------
def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Calcola metriche dettagliate. Fondamentale per vedere se il modello
    sta imparando le classi rare (Hybrid/Adversarial).
    """
    preds = np.array(preds)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "f1": float(f1),         
        "precision": float(precision),
        "recall": float(recall)
    }

    _, _, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    for i, score in enumerate(f1_per_class):
        metrics[f"f1_class_{i}"] = float(score)

    return metrics

# -----------------------------------------------------------------------------
# Evaluation Loop
# -----------------------------------------------------------------------------
def evaluate(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device,
    verbose: bool = False
) -> Tuple[Dict[str, float], List[int], List[int]]:
    
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
            lang_ids       = batch["lang_ids"].to(device, non_blocking=True)
            
            with autocast(device_type=device_type, dtype=dtype):
                logits, loss = model.forward(
                    input_ids, 
                    attention_mask, 
                    lang_ids=lang_ids,
                    labels=labels, 
                    alpha=0.0 
                )
            
            if loss is not None:
                running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.detach().cpu().numpy())
            references.extend(labels.detach().cpu().numpy())
            
            del input_ids, attention_mask, labels, lang_ids, logits, loss

    eval_metrics = compute_metrics(predictions, references)
    eval_metrics["loss"] = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    if verbose:
        report = classification_report(
            references, predictions, 
            zero_division=0,
            digits=4
        )
        logger.info(f"\nClassification Report:\n{report}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    return eval_metrics, predictions, references