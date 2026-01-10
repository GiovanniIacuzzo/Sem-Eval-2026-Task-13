import torch
import numpy as np
import logging
from typing import Dict, Tuple
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report
)
import random
import os
from torch.amp import autocast

logger = logging.getLogger(__name__)

# =============================================================================
# 1. UTILITIES DI BASE
# =============================================================================
class ConsoleUX:
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        log_str = f"[{stage}] "
        keys = ["loss", "f1_macro", "accuracy"]
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

# =============================================================================
# 2. METRICHE & VALUTAZIONE
# =============================================================================
def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    accuracy = accuracy_score(labels, preds)
    
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    
    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
    }
    
    if len(np.unique(labels)) <= 2:
        try:
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            metrics.update({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
        except Exception:
            pass 

    return metrics

def evaluate_model(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device,
    desc: str = "Evaluating"
) -> Tuple[Dict[str, float], str]:
    model.eval()
    
    loss_accum = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True)
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            struct_feats = batch.get("structural_features", None)
            if struct_feats is not None:
                struct_feats = struct_feats.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    structural_features=struct_feats,
                    labels=labels
                )
                
                loss = outputs["loss"]

            loss_accum += loss.item()
            
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            if progress_bar.n % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = loss_accum / len(dataloader)
    
    np_preds = torch.cat(all_preds).numpy()
    np_labels = torch.cat(all_labels).numpy()
    
    metrics = compute_metrics(np_preds, np_labels)
    metrics["loss"] = avg_loss
    
    target_names = ["Human (0)", "AI (1)"]
    try:
        report_str = classification_report(
            np_labels, np_preds, target_names=target_names, digits=4, zero_division=0
        )
    except Exception as e:
        report_str = f"Classification Report Error: {e}"
    
    return metrics, report_str