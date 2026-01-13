import torch
import numpy as np
import gc
import logging
import random
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from torch.amp import autocast
from tqdm import tqdm

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 0. General Utils
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """Fissa il seed per la riproducibilità su CPU, GPU e Numpy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# 1. Metric Computation
# -----------------------------------------------------------------------------
def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Calcola metriche dettagliate.
    Macro F1 è la metrica ufficiale per classi sbilanciate.
    """
    preds = np.array(preds)
    labels = np.array(labels)

    if len(labels) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0}

    accuracy = accuracy_score(labels, preds)
    
    precision_mac, recall_mac, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),         
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_mac),
        "recall_macro": float(recall_mac)
    }

    unique_labels = np.unique(np.concatenate([labels, preds]))
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=unique_labels, zero_division=0
    )

    for cls_idx, score in zip(unique_labels, f1_per_class):
        metrics[f"f1_class_{cls_idx}"] = float(score)

    return metrics

# -----------------------------------------------------------------------------
# 2. Evaluation Loop
# -----------------------------------------------------------------------------
def evaluate(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device,
    verbose: bool = False,
    label_names: Optional[List[str]] = None
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
            extra_features = batch["extra_features"].to(device, non_blocking=True)
            
            with autocast(device_type=device_type, dtype=dtype):
                logits, loss, _ = model(
                    input_ids, 
                    attention_mask, 
                    labels=labels, 
                    extra_features=extra_features
                )
            
            if loss is not None:
                running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.detach().cpu().numpy())
            references.extend(labels.detach().cpu().numpy())
            
            del input_ids, attention_mask, extra_features, labels, logits, loss

    eval_metrics = compute_metrics(predictions, references)
    eval_metrics["loss"] = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    if verbose:
        unique_labels = sorted(list(set(references) | set(predictions)))
        
        target_names = None
        if label_names is not None:
            if len(label_names) >= len(unique_labels):
                 target_names = [label_names[i] for i in unique_labels]
            else:
                 target_names = label_names

        try:
            logger.info("\n" + classification_report(
                references, predictions, 
                target_names=target_names,
                zero_division=0,
                digits=4
            ))
            cm = confusion_matrix(references, predictions)
            logger.info(f"Confusion Matrix:\n{cm}")
        except Exception as e:
            logger.warning(f"Could not print classification report: {e}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    return eval_metrics, predictions, references