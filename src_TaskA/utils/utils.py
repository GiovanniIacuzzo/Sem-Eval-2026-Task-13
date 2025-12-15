import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import autocast
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Metric Computation
# -----------------------------------------------------------------------------
def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Computes classification metrics with a focus on SemEval requirements.
    
    Args:
        preds (List[int]): List of predicted class indices.
        labels (List[int]): List of ground truth class indices.
        
    Returns:
        Dict[str, float]: Dictionary containing accuracy, precision, recall, and F1.
    """
    preds = np.array(preds)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, preds)
    
    # NOTE: SemEval Task 13 explicitly requires Macro F1-Score.
    # 'macro' calculates metrics for each label, and finds their unweighted mean.
    # This does not take label imbalance into account, treating all classes equally.
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
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
    Runs the evaluation loop on the provided dataloader.
    
    Features:
    - Mixed Precision Inference (MPS/CUDA) for reduced latency.
    - Non-blocking data transfer for pipeline optimization.
    
    Args:
        model: The PyTorch model wrapper.
        dataloader: Validation or Test dataloader.
        device: Execution device (CPU, MPS, or CUDA).
        
    Returns:
        Tuple containing:
        - metrics (Dict): Aggregated performance metrics.
        - predictions (List): Raw predictions.
        - references (List): Ground truth labels.
    """
    model.eval()
    running_loss = 0.0
    predictions = []
    references = []

    # 'leave=False' keeps the console clean by clearing the bar after completion
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False, dynamic_ncols=True)

    # Determine device type for autocast (portability between Mac M2 and NVIDIA GPUs)
    device_type = device.type if device.type in ['cuda', 'mps'] else 'cpu'
    # Use float16 for GPU/MPS acceleration, float32 (bfloat16) for CPU if needed
    dtype = torch.float16 if device_type != 'cpu' else torch.float32

    with torch.no_grad():
        for batch in progress_bar:
            # Non-blocking transfer allows asynchronous data movement
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)
            
            # Note: 'lang_ids' are omitted to force the model to rely solely on code structure,
            # enhancing Out-Of-Distribution (OOD) generalization.
            
            # Mixed Precision Context
            with autocast(device_type=device_type, dtype=dtype):
                logits, loss = model.forward(input_ids, attention_mask, labels=labels)
            
            if loss is not None:
                running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            
            # Move to CPU immediately to free up VRAM
            predictions.extend(preds.detach().cpu().numpy())
            references.extend(labels.detach().cpu().numpy())

    # Compute final aggregate metrics
    eval_metrics = compute_metrics(predictions, references)
    
    # Handle edge case for empty dataloader
    eval_metrics["loss"] = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    return eval_metrics, predictions, references