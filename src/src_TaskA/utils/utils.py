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

logger = logging.getLogger(__name__)


class DynamicCollate:
    """
    Gestisce il padding dinamico e l'impacchettamento delle Extra Features.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        
        # --- FIX: Gestione Extra Features ---
        extra_features = None
        # Controlliamo se il primo elemento ha 'extra_features' e se non è None
        if 'extra_features' in batch[0] and batch[0]['extra_features'] is not None:
            # Stack converte una lista di tensori in un unico tensore [Batch, Dim]
            extra_features = torch.stack([item['extra_features'] for item in batch])
        
        # Padding intelligente
        padded_inputs = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True, 
            return_tensors="pt"
        )

        return {
            "input_ids": padded_inputs["input_ids"],
            "attention_mask": padded_inputs["attention_mask"],
            "labels": labels,
            "extra_features": extra_features
        }

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calcola un set completo di metriche per la classificazione binaria/multiclasse.
    """
    # Calcolo metriche base
    accuracy = accuracy_score(labels, preds)
    
    # average='macro': tratta tutte le classi ugualmente (importante per OOD/classi rare)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    
    # average='weighted': pesa in base al numero di campioni (importante per il bilanciamento)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
    }
    
    # Aggiungiamo matrice di confusione se è binaria (0 vs 1)
    if len(np.unique(labels)) <= 2:
        try:
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            metrics.update({
                "TN": int(tn), "FP": int(fp), 
                "FN": int(fn), "TP": int(tp)
            })
        except ValueError:
            pass # Gestisce casi rari in cui manca una classe nel batch

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
            
            extra_features = batch.get("extra_features", None)
            if extra_features is not None:
                extra_features = extra_features.to(device, non_blocking=True)
            
            # Passiamo extra_features al modello
            outputs = model(input_ids, attention_mask, labels=labels, extra_features=extra_features)
            
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            loss_accum += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({"Val Loss": f"{loss.item():.4f}"})

    avg_loss = loss_accum / len(dataloader)
    
    np_preds = np.array(all_preds)
    np_labels = np.array(all_labels)
    
    metrics = compute_metrics(np_preds, np_labels)
    metrics["loss"] = avg_loss
    
    target_names = ["Human (0)", "AI (1)"]
    report_str = classification_report(
        np_labels, np_preds, target_names=target_names, digits=4, zero_division=0
    )
    
    return metrics, report_str