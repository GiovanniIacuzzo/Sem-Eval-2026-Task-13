import torch
import numpy as np
import logging
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score
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
def evaluate_model(model, dataloader, device, label_names=None):
    model.eval()
    loss_accum = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            extra_features = batch.get("extra_features", None)
            if extra_features is not None:
                extra_features = extra_features.to(device)
            
            logits, loss = model(
                input_ids, 
                attention_mask, 
                extra_features=extra_features,
                labels=labels, 
                alpha=0.0
            )
            
            loss_accum += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "loss": loss_accum / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted")
    }
    
    report = ""
    if label_names:
        try:
            report = classification_report(all_labels, all_preds, target_names=label_names, digits=4)
        except Exception as e:
            logger.warning(f"Classification Report Warning: {e}")
            
    return metrics, all_preds, all_labels, report