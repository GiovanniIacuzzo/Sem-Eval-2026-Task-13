import torch
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def set_seed(seed: int):
    """Fissa il seed per la riproducibilità."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(model, dataloader, device, label_names=None):
    """Funzione di valutazione robusta per Task A."""
    model.eval()
    preds_all = []
    labels_all = []
    total_loss = 0.0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            feats = batch["extra_features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # Forward (senza passare labels per ottenere logits puri per metrics)
            logits, _, _ = model(input_ids, mask, feats, labels=None)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu().numpy())
            
    # Metriche
    accuracy = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average='macro')
    # Gestione etichette mancanti nel report
    unique_labels = sorted(list(set(labels_all)))
    target_names = [label_names[i] for i in unique_labels] if label_names else None
    
    report = classification_report(labels_all, preds_all, target_names=target_names, digits=4, zero_division=0)
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "f1_macro": f1
    }, preds_all, labels_all, report