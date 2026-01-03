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
from transformers import AutoTokenizer, logging as transformers_logging

transformers_logging.set_verbosity_error()

from src.src_TaskA.models.model import CodeClassifier
from src.src_TaskA.dataset.dataset import CodeDataset, load_data_raw, balance_dataframe, get_dann_class_weights
from src.src_TaskA.utils.utils import evaluate, set_seed

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConsoleUX:    
    @staticmethod
    def header(fold, total_folds, val_lang):
        """Header per l'inizio di un nuovo Fold LOLO."""
        print(f"\n{'='*80}")
        print(f" FOLD {fold+1}/{total_folds} | VALIDATION LANGUAGE: {val_lang.upper()} (OOD TEST)")
        print(f"{'='*80}")

    @staticmethod
    def log_metrics(epoch, metrics, train_loss):
        """Log riassuntivo delle metriche a fine epoca."""
        logger.info(f"--- Epoch {epoch+1} Summary (Out-of-Distribution Validation) ---")
        logger.info(f" > Loss:        Train {train_loss:.4f} | Val {metrics['loss']:.4f}")
        logger.info(f" > Performance: F1 {metrics['f1']:.4f} (Threshold: {metrics['best_threshold']:.2f})")
        logger.info(f" > Class F1:    Human {metrics['human_f1']:.4f} | Machine {metrics['machine_f1']:.4f}")
        logger.info(f" > Detail (M):  Prec {metrics['precision_machine']:.4f} | Rec {metrics['recall_machine']:.4f}")
        print("-" * 40)

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
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        lang_ids = batch["lang_ids"].to(device, non_blocking=True)
        
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
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "alpha": f"{alpha:.2f}"})
        
    return running_loss / len_dl

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    set_seed(42)
    
    with open("src/src_TaskA/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    
    df_train_raw, df_val_raw = load_data_raw(config)
    full_df = pd.concat([df_train_raw, df_val_raw]).reset_index(drop=True)
    
    languages = sorted(full_df['language'].unique())
    language_map = {l: i for i, l in enumerate(languages)}
    config["model"]["num_languages"] = len(language_map)
    
    logger.info(f"Initialized LOLO Strategy on languages: {languages}")

    for fold, val_lang in enumerate(languages):
        ConsoleUX.header(fold, len(languages), val_lang)
        
        train_df_raw = full_df[full_df['language'] != val_lang]
        val_df = full_df[full_df['language'] == val_lang]
        
        train_df = balance_dataframe(train_df_raw, config)
        
        logger.info(f"Training on: {[l for l in languages if l != val_lang]}")
        logger.info(f"Dataset Sizes -> Train: {len(train_df)} | Val (OOD): {len(val_df)}")

        train_ds = CodeDataset(train_df, tokenizer, language_map, augment=True)
        val_ds = CodeDataset(val_df, tokenizer, language_map, augment=False)
        
        train_dl = DataLoader(
            train_ds, 
            batch_size=config["training"]["batch_size"], 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
        val_dl = DataLoader(
            val_ds, 
            batch_size=config["training"]["batch_size"]*2, 
            shuffle=False, 
            num_workers=4
        )
        
        dann_weights = get_dann_class_weights(train_df, language_map, device)
        model = CodeClassifier(config, dann_lang_weights=dann_weights).to(device)

        base_lr = float(config["training"]["learning_rate"])
        optimizer = torch.optim.AdamW([
            {'params': model.base_model.parameters(), 'lr': base_lr / 10},
            {'params': [p for n, p in model.named_parameters() if 'base_model' not in n], 'lr': base_lr}
        ], weight_decay=0.1)

        total_steps = (len(train_dl) // config["training"]["grad_accum_steps"]) * config["training"]["num_epochs"]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=base_lr, 
            total_steps=total_steps, 
            pct_start=0.1
        )
        scaler = GradScaler()

        best_f1 = 0
        patience_counter = 0
        checkpoint_dir = config["training"]["checkpoint_dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)

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
                save_path = os.path.join(checkpoint_dir, f"best_lolo_{val_lang}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'threshold': metrics['best_threshold'],
                    'config': config,
                    'f1': best_f1,
                    'val_lang': val_lang
                }, save_path)
                logger.info(f"New Best OOD F1: {best_f1:.4f} - Modello salvato per {val_lang}")
            else:
                patience_counter += 1
                if patience_counter >= config["training"].get("patience", 3):
                    logger.info(f"Early stopping triggerato al Fold {val_lang}")
                    break

        del model, optimizer, scheduler, train_dl, val_dl
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Pipeline LOLO completata. Modelli pronti per l'inferenza finale.")