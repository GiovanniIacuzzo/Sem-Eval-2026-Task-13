import os
import sys
import logging
import yaml
import torch
import numpy as np
import gc
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from comet_ml import Experiment

from src.src_TaskA.models.model import FusionCodeClassifier
from src.src_TaskA.dataset.dataset import CodeDataset, BalancedBatchSampler, load_full_data_for_kfold
from src.src_TaskA.utils.utils import evaluate

# -----------------------------------------------------------------------------
# Setup & Config
# -----------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConsoleUX:
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

# -----------------------------------------------------------------------------
# Training Loop Singola Epoca
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch_idx):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Ep {epoch_idx+1}", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        stylo_feats = batch.get("stylo_feats")
        labels = batch["labels"].to(device, non_blocking=True)
        
        if stylo_feats is not None:
            stylo_feats = stylo_feats.to(device, non_blocking=True)

        with autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(
                input_ids, attention_mask, stylo_feats=stylo_feats, labels=labels
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if scheduler:
            scheduler.step()
        
        running_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return running_loss / len(dataloader)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    ConsoleUX.print_banner("Task A - K-Fold Fusion Training")
    
    # Config Setup
    with open("src/src_TaskA/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    DATA_ROOT = os.getenv("DATA_PATH", "./data/Task_A")
    config["data"]["train_path"] = os.path.join(DATA_ROOT, "train.parquet")
    config["data"]["val_path"] = os.path.join(DATA_ROOT, "validation.parquet")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Caricamento Dati Unificato
    full_df = load_full_data_for_kfold(config)
    y_all = full_df['label'].values
    
    # Mappa lingue dinamica
    unique_langs = sorted(full_df['language'].unique().astype(str))
    language_map = {l: i for i, l in enumerate(unique_langs)}
    if 'unknown' not in language_map: language_map['unknown'] = len(language_map)

    # Tokenizer
    model_name = config["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # K-Fold Setup
    k_folds = 5
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Comet Experiment
    experiment = None
    if os.getenv("COMET_API_KEY"):
        experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name="semeval-task-a-fusion",
            auto_metric_logging=False
        )
        experiment.log_parameters(config)

    # -------------------------------------------------------------------------
    # LOOP DEI FOLD
    # -------------------------------------------------------------------------
    results_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(y_all)), y_all)):
        ConsoleUX.print_banner(f"FOLD {fold+1}/{k_folds}")
        
        # Split Dataframe
        train_df_fold = full_df.iloc[train_idx].reset_index(drop=True)
        val_df_fold = full_df.iloc[val_idx].reset_index(drop=True)
        
        # Dataset Creation
        train_dataset = CodeDataset(train_df_fold, tokenizer, language_map, 
                                   max_length=config["data"]["max_length"], augment=True)
        val_dataset = CodeDataset(val_df_fold, tokenizer, language_map, 
                                 max_length=config["data"]["max_length"], augment=False)
        
        train_sampler = BalancedBatchSampler(
            train_dataset.labels_list, 
            batch_size=config["training"]["batch_size"]
        )
        
        train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, 
                             num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_dataset, batch_size=config["training"]["batch_size"]*2, 
                           shuffle=False, num_workers=4, pin_memory=True)
        
        logger.info(f"Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

        # Init Model
        model = FusionCodeClassifier(config)
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
        scaler = GradScaler()
        
        num_epochs = config["training"]["num_epochs"]
        steps_per_epoch = len(train_dl)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=float(config["training"]["learning_rate"]),
            total_steps=steps_per_epoch * num_epochs, pct_start=0.1
        )

        best_f1 = 0.0
        fold_save_path = os.path.join(config["training"]["checkpoint_dir"], f"fold_{fold}")
        os.makedirs(fold_save_path, exist_ok=True)

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_dl, optimizer, scheduler, scaler, device, epoch)
            
            # Valutazione
            val_metrics, _, _ = evaluate(model, val_dl, device)
            
            logger.info(f"Ep {epoch+1} | Loss: {train_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
            
            if experiment:
                experiment.log_metrics(val_metrics, prefix=f"Fold{fold}_Val", step=epoch)
            
            # Save best
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save(model.state_dict(), os.path.join(fold_save_path, "best_model.pt"))
                logger.info("--> New Best Model Saved")

        results_per_fold.append(best_f1)
        
        # Cleanup Memory
        del model, optimizer, scheduler, scaler, train_dl, val_dl, train_dataset, val_dataset
        torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"\nTraining Completed. Average F1 across folds: {np.mean(results_per_fold):.4f}")