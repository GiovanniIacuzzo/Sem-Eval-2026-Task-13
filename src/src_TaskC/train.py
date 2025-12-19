import os
import sys
import logging
import yaml
import torch
import numpy as np
import zipfile
import shutil
import argparse
import gc
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from comet_ml import Experiment

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Warning: 'kaggle' library not found. Auto-download disabled.")

# --- IMPORTS AGGIORNATI ---
from src_TaskC.models.model import CodeClassifier
from src_TaskC.dataset.dataset import load_data
from src_TaskC.utils.utils import evaluate

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConsoleUX:
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        # Ordina le chiavi per avere f1 globale prima delle classi
        keys = sorted(metrics.keys(), key=lambda x: (0 if x == 'f1' else 1, x))
        log_str = f"[{stage}] "
        for k in keys:
            v = metrics[k]
            if "class" in k: # Accorcia i log per le classi
                log_str += f"{k.replace('f1_class_', 'C')}: {v:.3f} | "
            elif isinstance(v, float):
                log_str += f"{k}: {v:.4f} | "
            else:
                log_str += f"{k}: {v} | "
        logger.info(log_str.strip(" | "))

# -----------------------------------------------------------------------------
# Kaggle Data Helper
# -----------------------------------------------------------------------------
def setup_kaggle_data(download_dir="./data"):
    COMPETITION_SLUG = "semeval-2026-task-13-subtask-c" 
    task_dir = os.path.join(download_dir, "Task_C") 
    train_file = os.path.join(task_dir, "train.parquet")
    
    if os.path.exists(train_file):
        logger.info(f"Data found in {task_dir}. Skipping download.")
        return task_dir

    if not KAGGLE_AVAILABLE:
        logger.error("Kaggle lib not installed & data missing. Please download manually.")
        sys.exit(1)

    logger.info(f"Downloading data from Kaggle: {COMPETITION_SLUG}...")
    try:
        api = KaggleApi()
        api.authenticate()
        os.makedirs(task_dir, exist_ok=True)
        api.competition_download_files(COMPETITION_SLUG, path=task_dir, quiet=False)
        for item in os.listdir(task_dir):
            if item.endswith(".zip"):
                zip_path = os.path.join(task_dir, item)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(task_dir)
                os.remove(zip_path)
        return task_dir
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        raise e

# -----------------------------------------------------------------------------
# Training Routine Single Epoch
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, 
                   epoch_idx, total_epochs, accumulation_steps=1):
    
    model.train()
    running_loss = 0.0
    predictions, references = [], []
    
    optimizer.zero_grad()
    
    len_dataloader = len(dataloader)
    progress_bar = tqdm(dataloader, desc=f"Ep {epoch_idx+1} Train", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        lang_ids = batch["lang_ids"].to(device, non_blocking=True)
        
        # Annealing per DANN (Gradient Reversal)
        current_step = step + epoch_idx * len_dataloader
        total_steps = total_epochs * len_dataloader
        p = float(current_step) / total_steps
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        with autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(
                input_ids, attention_mask, 
                lang_ids=lang_ids, labels=labels, alpha=alpha
            )
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        current_loss = loss.item() * accumulation_steps
        running_loss += current_loss
        
        progress_bar.set_postfix({
            "Loss": f"{current_loss:.4f}", 
            "LR": f"{scheduler.get_last_lr()[0]:.1e}"
        })

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        predictions.extend(preds)
        references.extend(labels_cpu)
        
        # Memory cleanup
        del input_ids, attention_mask, labels, lang_ids, logits, loss
    
    # Compute metrics at end of epoch
    metrics = model.compute_metrics(predictions, references)
    metrics["loss"] = running_loss / len_dataloader
    return metrics

# -----------------------------------------------------------------------------
# Main Execution (K-FOLD ENABLED)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    ConsoleUX.print_banner("SemEval 2026 Task 13 - Subtask C (K-Fold Optimized)")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src_TaskC/config/config.yaml") 
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    DATA_ROOT = os.getenv("DATA_PATH", "./data")
    task_data_dir = setup_kaggle_data(DATA_ROOT) 
    
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Config not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["data"]["train_path"] = os.path.join(task_data_dir, "train.parquet")
    config["data"]["val_path"] = os.path.join(task_data_dir, "validation.parquet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # 1. Caricamento Dati Completo (Uniamo train e val per la Cross Validation)
    # Passiamo 'device' per calcolare i pesi direttamente su GPU/CPU
    # NOTA: Usiamo il modello temporaneo solo per il tokenizer
    temp_tokenizer = CodeClassifier(config).tokenizer 
    full_train_dataset, full_val_dataset, class_weights = load_data(config, temp_tokenizer, device)

    # Uniamo i dataset per K-Fold
    full_dataset = torch.utils.data.ConcatDataset([full_train_dataset, full_val_dataset])
    # Estraiamo le labels per StratifiedKFold
    all_labels = []
    # Attenzione: Iterare su ConcatDataset è lento, estraiamo labels dal dataframe originale
    # Assumiamo che load_data restituisca anche i df o che li possiamo ricavare.
    # Per semplicità qui usiamo gli indici, ma in produzione meglio passare y a KFold.
    # Recuperiamo y dal dataframe interno dei dataset
    y_train = full_train_dataset.data['label'].values
    y_val = full_val_dataset.data['label'].values
    y_all = np.concatenate([y_train, y_val])

    # K-Fold Setup
    kf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    
    # Init CometML Experiment (Global)
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME", "semeval-task13-subtaskc"),
        workspace=os.getenv("COMET_WORKSPACE"),
        auto_metric_logging=False
    )
    experiment.log_parameters(config)

    # -------------------------------------------------------------------------
    # K-Fold Loop
    # -------------------------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(y_all)), y_all)):
        ConsoleUX.print_banner(f"FOLD {fold+1}/{args.k_folds}")
        
        # Re-inizializza il modello per ogni fold
        model = CodeClassifier(config, class_weights=class_weights)
        model.to(device)
        
        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)
        
        num_workers = 4
        train_dl = DataLoader(train_sub, batch_size=config["training"]["batch_size"], 
                             shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        val_dl = DataLoader(val_sub, batch_size=config["training"]["batch_size"], 
                           shuffle=False, num_workers=num_workers, pin_memory=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=0.01)
        scaler = GradScaler()
        
        acc_steps = config["training"].get("gradient_accumulation_steps", 1)
        num_epochs = config["training"].get("num_epochs", 5) # Default 5 epoch per fold
        total_steps = len(train_dl) * num_epochs // acc_steps
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=float(config["training"]["learning_rate"]),
            total_steps=total_steps, pct_start=0.1
        )

        best_fold_f1 = 0.0
        patience = config["training"].get("early_stop_patience", 3)
        patience_counter = 0
        save_dir = config["training"]["checkpoint_dir"]
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            train_metrics = train_one_epoch(
                model, train_dl, optimizer, scheduler, scaler, device, 
                epoch, num_epochs, acc_steps
            )
            
            # Valutazione con Report Verbose all'ultima epoch o se migliora
            val_metrics, val_preds, val_refs = evaluate(model, val_dl, device, verbose=False)
            
            ConsoleUX.log_metrics(f"F{fold+1}-Ep{epoch+1}", val_metrics)
            
            # Logging to Comet (con prefisso fold)
            experiment.log_metrics(train_metrics, prefix=f"Fold{fold}_Train", step=epoch)
            experiment.log_metrics(val_metrics, prefix=f"Fold{fold}_Val", step=epoch)

            if val_metrics["f1"] > best_fold_f1:
                logger.info(f"⭐ New Best F1 for Fold {fold+1}: {val_metrics['f1']:.4f}")
                best_fold_f1 = val_metrics["f1"]
                patience_counter = 0
                
                # Salvataggio specifico per Fold
                fold_save_dir = os.path.join(save_dir, f"fold_{fold}")
                os.makedirs(fold_save_dir, exist_ok=True)
                
                if model.use_lora:
                    model.base_model.save_pretrained(fold_save_dir)
                    torch.save(model.classifier.state_dict(), os.path.join(fold_save_dir, "head.pt"))
                else:
                    torch.save(model.state_dict(), os.path.join(fold_save_dir, "model.pt"))
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping fold {fold+1}")
                break

        # Cleanup tra i fold per liberare VRAM
        del model, optimizer, scheduler, scaler, train_dl, val_dl
        torch.cuda.empty_cache()
        gc.collect()

    experiment.end()
    logger.info("Training Finished. All folds processed.")