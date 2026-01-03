import torch
import numpy as np
import gc
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_recall_curve, classification_report
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
def evaluate(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    all_probs = []
    all_labels = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                # (loss, logits)
                _, logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            
            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = val_loss / len(dataloader)
    
    # --- THRESHOLD TUNING ---
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1_val = np.max(f1_scores)
    
    final_preds = (all_probs >= best_threshold).astype(int)
    
    report = classification_report(all_labels, final_preds, target_names=["Human", "Machine"], output_dict=True, zero_division=0)
    
    metrics = {
        "loss": avg_loss,
        "acc": report["accuracy"],
        "f1": best_f1_val,
        "best_threshold": best_threshold,
        "human_f1": report["Human"]["f1-score"],
        "machine_f1": report["Machine"]["f1-score"],
        "precision_machine": report["Machine"]["precision"],
        "recall_machine": report["Machine"]["recall"]
    }
    
    return metrics