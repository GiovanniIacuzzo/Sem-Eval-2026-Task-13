import torch
import numpy as np
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import autocast
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Metric Computation Strategy
# -----------------------------------------------------------------------------
def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Computes classification metrics aligned with SemEval competition standards.
    
    Why Macro F1?
    The competition datasets may have class imbalances (e.g., more Human code than AI).
    Macro-averaging calculates metrics independently for each class and then takes 
    the unweighted mean, treating all classes as equally important regardless of frequency.
    
    Args:
        preds (List[int]): List of predicted class indices.
        labels (List[int]): List of ground truth class indices.
        
    Returns:
        Dict[str, float]: Dictionary containing Accuracy, Precision, Recall, and F1 (Macro).
    """
    preds = np.array(preds)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, preds)
    
    # Calculate Precision, Recall, and F1
    # zero_division=0 prevents runtime warnings if a class is never predicted
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
# Inference & Evaluation Loop
# -----------------------------------------------------------------------------
def evaluate(model, dataloader, device):
    """
    Validation loop optimized for inference.
    """
    model.eval()
    predictions, references = [], []
    running_loss = 0.0
    
    # Mixed precision context for inference too (speedup on M2)
    device_type = device.type if device.type in ['cuda', 'mps'] else 'cpu'
    dtype = torch.float16 if device_type != 'cpu' else torch.float32

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast(device_type=device_type, dtype=dtype):
                logits, loss = model(input_ids, attention_mask, labels=labels)

            if loss is not None:
                running_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            
            predictions.extend(preds)
            references.extend(labels_cpu)
            
            # Memory housekeeping
            del input_ids, attention_mask, labels, logits, loss

    metrics = model.compute_metrics(predictions, references)
    metrics["loss"] = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    return metrics, predictions, references