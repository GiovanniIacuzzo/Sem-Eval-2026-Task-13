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
# Metric Computation Strategy
# -----------------------------------------------------------------------------
def compute_metrics(preds: List[int], labels: List[int], label_names: List[str] = None) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics.
    Returns a dictionary with scalar metrics (for logging) and a text report (for console).
    """
    preds = np.array(preds)
    labels = np.array(labels)

    # 1. Metriche Globali
    accuracy = accuracy_score(labels, preds)
    
    # Macro: Media aritmetica delle metriche per classe (Target SemEval)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    # Weighted: Media pesata per support (Utile per capire la performance reale sui dati)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )

    # 2. Metriche Per Classe (Cruciale per il debug)
    # Calcoliamo F1 per ogni singola classe presente
    unique_labels = np.unique(labels)
    p_per_cls, r_per_cls, f1_per_cls, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=unique_labels, zero_division=0
    )
    
    # Costruiamo il dizionario risultati
    results = {
        "accuracy": float(accuracy),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),          # <-- TARGET PRINCIPALE
        "f1_weighted": float(f1_weighted)
    }

    # Aggiungiamo F1 per classe al dizionario (es. "f1_cls_0": 0.85)
    for i, label_idx in enumerate(unique_labels):
        label_name = label_names[label_idx] if label_names and label_idx < len(label_names) else f"cls_{label_idx}"
        results[f"f1_{label_name}"] = float(f1_per_cls[i])

    # 3. Report Testuale Completo (per i log da console)
    # Se passiamo i nomi delle label, il report è molto più leggibile
    target_names = None
    if label_names:
        # Filtriamo i nomi solo per le classi presenti nel batch/epoch (per evitare warning sklearn)
        # O passiamo tutti se siamo sicuri che le label siano 0..N-1
        max_label = max(labels.max(), preds.max())
        if max_label < len(label_names):
             target_names = label_names[:max_label+1]
    
    report_str = classification_report(
        labels, 
        preds, 
        target_names=target_names, 
        zero_division=0,
        digits=4 # Più precisione decimale
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
    
    # Nota: su T4/Colab, 'deterministic=True' può rallentare molto il training.
    # 'benchmark=True' ottimizza le convoluzioni per la velocità.
    # Per competizioni dove il tempo conta, questa config è il compromesso migliore:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Global seed set to: {seed}")

# -----------------------------------------------------------------------------
# Inference & Evaluation Loop (Optimized for T4/CUDA)
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
    
    # Rilevamento device type e dtype robusto per Autocast
    if device.type == 'cuda':
        device_type = 'cuda'
        dtype = torch.float16
    elif device.type == 'mps': # Mac M1/M2/M3
        device_type = 'mps'
        dtype = torch.float16
    else:
        device_type = 'cpu'
        dtype = torch.bfloat16 # bfloat16 è meglio su CPU moderne, altrimenti float32
        
    # Disabilita il calcolo dei gradienti per risparmiare VRAM
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            # Spostamento su GPU
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # Gestione Lang IDs (per DANN)
            # In validazione il DANN deve essere spento (alpha=0), ma il modello
            # potrebbe aspettarsi comunque il tensore per il forward pass tecnico.
            lang_ids = batch.get("lang_ids", None)
            if lang_ids is not None:
                lang_ids = lang_ids.to(device, non_blocking=True)

            # Inference context
            with autocast(device_type=device_type, dtype=dtype):
                # IMPORTANTE: alpha=0.0 disabilita l'Adversarial Training.
                # In validazione vogliamo features pulite, non gradient reversal.
                logits, loss = model(
                    input_ids, 
                    attention_mask, 
                    lang_ids=lang_ids, 
                    labels=labels, 
                    alpha=0.0 
                )

            if loss is not None:
                running_loss += loss.item()

            # Move to CPU for metrics
            # Argmax sui logits per ottenere la classe predetta
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()
            
            predictions.extend(preds)
            references.extend(labels_cpu)

    # Calcolo Loss Media
    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    # Calcolo Metriche
    # compute_metrics ora restituisce una tupla (dict_metriche, stringa_report)
    metrics, report_str = compute_metrics(predictions, references, label_names)
    metrics["loss"] = avg_loss
    
    # Logghiamo il report testuale solo a livello DEBUG o INFO se necessario
    # logger.info(f"\nValidation Report:\n{report_str}") 
    # (Lo faremo nel main loop per non sporcare questo file)

    # Pulizia memoria aggressiva per evitare OOM tra Validation e Train
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()
    
    return metrics, predictions, references, report_str