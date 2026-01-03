import torch
import numpy as np
import logging
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import random
import os

logger = logging.getLogger(__name__)

class ConsoleUX:
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        log_str = f"[{stage}] "
        keys = ["f1_macro", "f1_weighted", "accuracy", "loss"]
        for k in keys:
            if k in metrics:
                log_str += f"{k}: {metrics[k]:.4f} | "
        logger.info(log_str.strip(" | "))

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Global seed set to: {seed}")

def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, Any]:
    preds = np.array(preds)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    
    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro)
    }

def evaluate_model(model, dataloader, device, label_names=None):
    model.eval()
    loss_accum = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            extra_features = batch.get("extra_features", None)
            if extra_features is not None:
                extra_features = extra_features.to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, labels=labels, extra_features=extra_features)
            
            # Gestione output (logits, loss, features)
            if isinstance(outputs, tuple):
                logits = outputs[0]
                loss = outputs[1]
            else:
                logits = outputs.logits
                loss = outputs.loss
            
            loss_accum += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    final_loss = loss_accum / len(dataloader)
    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = final_loss
    
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=label_names, 
        digits=4, 
        zero_division=0
    )
    return metrics, all_preds, all_labels, report