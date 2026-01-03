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
from transformers import AutoTokenizer, logging as transformers_logging

transformers_logging.set_verbosity_error()

from src.src_TaskA.models.model import CodeClassifier
from src.src_TaskA.dataset.dataset import CodeDataset, load_data_raw, balance_dataframe, get_dann_class_weights
from src.src_TaskA.utils.utils import evaluate

# --- CONFIGURAZIONE LOGGING ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConsoleUX:    
    @staticmethod
    def header(fold, total_folds, model_name):
        """Header testuale semplice."""
        print(f"\n{'='*80}")
        print(f" FOLD {fold+1}/{total_folds} | MODEL: {model_name}")
        print(f"{'='*80}")

    @staticmethod
    def log_metrics(epoch, metrics, train_loss):
        """Log delle metriche in formato lista chiara."""
        logger.info(f"--- Epoch {epoch+1} Summary ---")
        logger.info(f" > Loss:        Train {train_loss:.4f} | Val {metrics['loss']:.4f}")
        logger.info(f" > Performance: F1 {metrics['f1']:.4f} (Thresh: {metrics['best_threshold']:.2f})")
        logger.info(f" > Class F1:    Human {metrics['human_f1']:.4f} | Machine {metrics['machine_f1']:.4f}")
        logger.info(f" > Detail (M):  Prec {metrics['precision_machine']:.4f} | Rec {metrics['recall_machine']:.4f}")
        print("-" * 30)

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch_idx, total_epochs, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    len_dl = len(dataloader)
    
    pbar = tqdm(
        dataloader, 
        desc=f"  Training Ep {epoch_idx+1}", 
        leave=False, 
        bar_format="{l_bar}{bar:30}{r_bar}"
    )
    
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        lang_ids = batch["lang_ids"].to(device)
        
        p = float(step + epoch_idx * len_dl) / (total_epochs * len_dl)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        with autocast(device_type='cuda', dtype=torch.float16):
            loss, _ = model(input_ids, attention_mask, lang_ids=lang_ids, labels=labels, alpha=alpha)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler: scheduler.step()
        
        running_loss += loss.item() * accumulation_steps
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "a": f"{alpha:.2f}"})
        
    return running_loss / len_dl

if __name__ == "__main__":
    load_dotenv()
    with open("src/src_TaskA/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    
    # Caricamento Dati
    df_train_raw, df_val_raw = load_data_raw(config)
    full_df = pd.concat([df_train_raw, df_val_raw]).reset_index(drop=True)
    
    language_map = {l: i for i, l in enumerate(sorted(full_df['language'].unique()))}
    config["model"]["num_languages"] = len(language_map)
    
    logger.info(f"Languages identified: {list(language_map.keys())}")
    
    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=config["training"]["k_folds"], shuffle=True, random_state=42)
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(full_df, full_df['label'])):
        ConsoleUX.header(fold, config["training"]["k_folds"], config["model"]["model_name"])
        
        train_df = balance_dataframe(full_df.iloc[t_idx], config)
        val_df = full_df.iloc[v_idx]
        
        logger.info(f"Samples: Train {len(train_df)} | Val {len(val_df)}")

        train_ds = CodeDataset(train_df, tokenizer, language_map, augment=True)
        val_ds = CodeDataset(val_df, tokenizer, language_map, augment=False)
        
        train_dl = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=config["training"]["batch_size"]*2, shuffle=False, num_workers=4)
        
        dann_weights = get_dann_class_weights(train_df, language_map, device)
        model = CodeClassifier(config, dann_lang_weights=dann_weights).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=0.05)
        
        total_steps = (len(train_dl) // config["training"]["grad_accum_steps"]) * config["training"]["num_epochs"]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=float(config["training"]["learning_rate"]),
            total_steps=total_steps, pct_start=0.1
        )
        scaler = GradScaler()

        best_f1 = 0
        patience_counter = 0

        for epoch in range(config["training"]["num_epochs"]):
            avg_train_loss = train_one_epoch(
                model, train_dl, optimizer, scheduler, scaler, device, 
                epoch, config["training"]["num_epochs"], config["training"]["grad_accum_steps"]
            )
            
            metrics = evaluate(model, val_dl, device)
            
            ConsoleUX.log_metrics(epoch, metrics, avg_train_loss)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                patience_counter = 0
                save_path = os.path.join(config["training"]["checkpoint_dir"], f"best_fold{fold}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'threshold': metrics['best_threshold'],
                    'config': config,
                    'f1': best_f1,
                    'language_map': language_map
                }, save_path)
                logger.info(f"*** New Best F1: {best_f1:.4f} - Model Saved ***\n")
            else:
                patience_counter += 1
                if patience_counter >= config["training"].get("patience", 4):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        del model, optimizer, scheduler, train_dl, val_dl
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Training process completed.")