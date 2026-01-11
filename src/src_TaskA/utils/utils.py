import torch
import random
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

def evaluate_model(model, dataloader, device):
    """
    Esegue l'inferenza sul validation set e calcola le metriche.
    Ritorna:
        - metrics (dict): {loss, accuracy, f1_macro}
        - report (str): Classification report testuale
    """
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            sem_emb = batch["semantic_embedding"].to(device, non_blocking=True)
            struct_feats = batch["structural_features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            outputs = model(sem_emb, struct_feats, labels)
            
            loss = outputs["loss"]
            total_loss += loss.item()
            
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calcolo metriche
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "f1_macro": f1
    }
    
    # Report testuale
    report = classification_report(all_labels, all_preds, target_names=["Human", "AI"], digits=4)
    
    return metrics, report