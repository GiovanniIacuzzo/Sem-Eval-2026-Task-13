import torch
import numpy as np
import logging
from typing import Dict, List, Tuple
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
    """
    Esegue il loop di valutazione completo.
    
    Args:
        model: Il modello PyTorch (CodeClassifier).
        dataloader: Il DataLoader di validazione.
        device: 'cuda' o 'cpu'.
        desc: Descrizione per la progress bar.
        
    Returns:
        metrics: Dizionario con i valori numerici.
        report_str: Stringa formattata (Classification Report) per loggare.
    """
    model.eval()
    
    loss_accum = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    
    # Disabilitiamo il calcolo dei gradienti per risparmiare memoria e calcolo
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True)
        
        for batch in progress_bar:
            # Spostamento dati su Device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, labels=labels)
            
            # Estrazione Loss e Logits
            # Nota: il modello ritorna un dict
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            loss_accum += loss.item()
            
            # Calcolo predizioni (Argmax)
            preds = torch.argmax(logits, dim=1)
            
            # Spostiamo su CPU subito per liberare VRAM
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Aggiorniamo la barra con la loss corrente
            progress_bar.set_postfix({"Val Loss": f"{loss.item():.4f}"})

    # Aggregazione finale
    avg_loss = loss_accum / len(dataloader)
    
    # Conversione in numpy array per sklearn
    np_preds = np.array(all_preds)
    np_labels = np.array(all_labels)
    
    # Calcolo metriche
    metrics = compute_metrics(np_preds, np_labels)
    metrics["loss"] = avg_loss
    
    # Generazione report testuale dettagliato
    target_names = ["Human (0)", "AI (1)"]
    report_str = classification_report(
        np_labels, 
        np_preds, 
        target_names=target_names, 
        digits=4, 
        zero_division=0
    )
    
    return metrics, report_str