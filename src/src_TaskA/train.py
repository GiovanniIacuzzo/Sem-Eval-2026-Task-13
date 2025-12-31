import os
import sys
import logging
import yaml
import torch
import gc
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

try:
    from comet_ml import Experiment
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

from src.src_TaskA.models.model import CodeClassifier
from src.src_TaskA.dataset.dataset import CodeDataset, load_data_raw, balance_dataframe, get_dann_class_weights
from src.src_TaskA.utils.utils import evaluate

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

class ConsoleUX:
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        keys = sorted(metrics.keys(), key=lambda x: (0 if x == 'f1' else 1, x))
        log_str = f"[{stage}] "
        for k in keys:
            v = metrics[k]
            if "class" in k: 
                log_str += f"{k.replace('f1_class_', 'C')}: {v:.3f} | "
            elif isinstance(v, float):
                log_str += f"{k}: {v:.4f} | "
            else:
                log_str += f"{k}: {v} | "
        logger.info(log_str.strip(" | "))

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch_idx, total_epochs, accumulation_steps=1, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    len_dataloader = len(dataloader)
    progress_bar = tqdm(dataloader, desc=f"Ep {epoch_idx+1} Train", leave=False)
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        lang_ids = batch["lang_ids"].to(device)
        
        p = float(step + epoch_idx * len_dataloader) / (total_epochs * len_dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        with autocast(device_type='cuda', dtype=torch.float16):
            loss, logits = model(input_ids, attention_mask, lang_ids=lang_ids, labels=labels, alpha=alpha)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None: scheduler.step()
        
        running_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix({"Loss": f"{loss.item()*accumulation_steps:.4f}"})
        
    return {"loss": running_loss / len_dataloader}

if __name__ == "__main__":
    load_dotenv()
    ConsoleUX.print_banner("SemEval 2026 - Task 13 - subtask A")
    
    with open("src/src_TaskA/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # --- DATA PREPARATION ---
    # 1. Carica dati GREZZI
    df_train_raw, df_val_raw = load_data_raw(config)
    
    # 2. Unisci per K-Fold
    full_df = pd.concat([df_train_raw, df_val_raw]).reset_index(drop=True)
    y_all = full_df['label'].values 
    
    # 3. Mappa lingue su tutto il dataset
    langs = sorted(full_df['language'].unique())
    language_map = {l: i for i, l in enumerate(langs)}
    config["model"]["num_languages"] = len(language_map)
    logger.info(f"Language Map ({len(langs)}): {language_map}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    
    # --- K-FOLD LOOP ---
    k_folds = config["training"].get("k_folds", 5)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config["experiment"]["seed"])
    
    base_checkpoint_dir = config["training"]["checkpoint_dir"]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all)):
        ConsoleUX.print_banner(f"FOLD {fold+1}/{k_folds}")
        
        # Split
        train_fold = full_df.iloc[train_idx]
        val_fold = full_df.iloc[val_idx]

        logger.info(f"Balancing Train Fold {fold+1}...")
        train_fold = balance_dataframe(train_fold, config)
        logger.info(f"Train Fold size: {len(train_fold)} | Val Fold size: {len(val_fold)}")

        # Datasets
        train_ds = CodeDataset(train_fold, tokenizer, language_map, augment=True)
        val_ds = CodeDataset(val_fold, tokenizer, language_map, augment=False)
        
        train_dl = DataLoader(train_ds, batch_size=config["training"]["batch_size"], 
                             shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=config["training"]["batch_size"]*2, 
                           shuffle=False, num_workers=4, pin_memory=True)
        
        dann_weights = get_dann_class_weights(train_fold, language_map, device)
        
        model = CodeClassifier(config, dann_lang_weights=dann_weights)
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=config["training"]["weight_decay"])
        scaler = GradScaler()
        
        num_epochs = config["training"]["num_epochs"]
        total_steps = len(train_dl) * num_epochs // config["training"]["grad_accum_steps"]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(config["training"]["learning_rate"]), total_steps=total_steps, pct_start=0.1)

        best_f1 = 0.0
        patience_counter = 0
        patience = config["training"]["patience"]
        
        # Experiment Logger
        experiment = None
        if COMET_AVAILABLE and os.getenv("COMET_API_KEY"):
            experiment = Experiment(
                api_key=os.getenv("COMET_API_KEY"),
                project_name=os.getenv("COMET_PROJECT_NAME", "semeval-taskA"),
                auto_metric_logging=False
            )
            experiment.log_parameters(config)
            experiment.add_tag(f"fold_{fold}")

        for epoch in range(num_epochs):
            train_metrics = train_one_epoch(
                model, train_dl, optimizer, scheduler, scaler, device, 
                epoch, num_epochs, config["training"]["grad_accum_steps"], config["training"]["max_grad_norm"]
            )
            val_metrics = evaluate(model, val_dl, device)

            ConsoleUX.log_metrics(f"F{fold+1}-Ep{epoch+1}", val_metrics)
            
            if experiment:
                experiment.log_metrics(train_metrics, prefix=f"fold{fold}_train", step=epoch)
                experiment.log_metrics(val_metrics, prefix=f"fold{fold}_val", step=epoch)
            
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                patience_counter = 0
                
                fold_dir = os.path.join(base_checkpoint_dir, f"fold_{fold}")
                os.makedirs(fold_dir, exist_ok=True)
                
                if hasattr(model, "base_model") and hasattr(model.base_model, "save_pretrained"):
                    model.base_model.save_pretrained(fold_dir)
                    torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))
                else:
                    torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping fold {fold+1}")
                    break
        
        if experiment: experiment.end()
        del model, optimizer, scheduler, scaler, train_dl, val_dl, train_ds, val_ds
        gc.collect()
        torch.cuda.empty_cache()
    
    logger.info("K-Fold Training Finished.")