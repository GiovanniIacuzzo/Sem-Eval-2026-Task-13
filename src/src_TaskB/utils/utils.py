import torch
import numpy as np
import gc
import logging
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.amp import autocast
from tqdm import tqdm
import random
import os

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Metric Computation
# -----------------------------------------------------------------------------
def compute_metrics(preds: List[int], labels: List[int], label_names: List[str] = None) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics.
    Returns a dictionary with scalar metrics (for logging) and a text report (for console).
    """
    preds = np.array(preds)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, preds)
    
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )

    unique_labels = np.unique(labels)
    p_per_cls, r_per_cls, f1_per_cls, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=unique_labels, zero_division=0
    )
    
    results = {
        "accuracy": float(accuracy),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted)
    }

    for i, label_idx in enumerate(unique_labels):
        label_name = label_names[label_idx] if label_names and label_idx < len(label_names) else f"cls_{label_idx}"
        results[f"f1_{label_name}"] = float(f1_per_cls[i])

    target_names = None
    if label_names:
        max_label = max(labels.max(), preds.max())
        if max_label < len(label_names):
             target_names = label_names[:max_label+1]
    
    report_str = classification_report(
        labels, 
        preds, 
        target_names=target_names, 
        zero_division=0,
        digits=4
    )
    
    return results, report_str

def set_seed(seed: int = 42):
    """
    Fissa il seed per tutte le librerie (Python, NumPy, PyTorch)
    per garantire la riproducibilità degli esperimenti.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Global seed set to: {seed}")

# -----------------------------------------------------------------------------
# Inference & Evaluation Loop
# -----------------------------------------------------------------------------
def evaluate(model, dataloader, device, label_names=None) -> Tuple[Dict[str, float], List[int], List[int]]:
    """
    Validation loop optimized for inference stability on CUDA.
    Args:
        label_names: Lista di stringhe con i nomi delle classi per il report dettagliato.
    """
    model.eval()
    predictions, references = [], []
    running_loss = 0.0
    
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
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            lang_ids = batch.get("lang_ids", None)
            if lang_ids is not None:
                lang_ids = lang_ids.to(device, non_blocking=True)

            with autocast(device_type=device_type, dtype=dtype):
                logits, loss = model(
                    input_ids, 
                    attention_mask, 
                    lang_ids=lang_ids, 
                    labels=labels, 
                    alpha=0.0 
                )

            if loss is not None:
                running_loss += loss.item()

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()
            
            predictions.extend(preds)
            references.extend(labels_cpu)

    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    metrics, report_str = compute_metrics(predictions, references, label_names)
    metrics["loss"] = avg_loss
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()
    
    return metrics, predictions, references, report_str