import os
import sys
import logging
import yaml
import torch
import gc
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RI_AVAILABLE = True
    console = Console()
except ImportError:
    RI_AVAILABLE = False

from src.src_TaskA.models.model import CodeClassifier
from src.src_TaskA.dataset.dataset import CodeDataset, load_data_raw, balance_dataframe, get_dann_class_weights
from src.src_TaskA.utils.utils import evaluate

# --- CONFIGURAZIONE LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConsoleUX:    
    @staticmethod
    def header(fold, total_folds, model_name):
        now = datetime.now().strftime("%H:%M:%S")
        text = f"FOLD {fold+1}/{total_folds} | {model_name} | {now}"
        if RI_AVAILABLE:
            console.print(Panel(text, style="bold magenta", expand=False))
        else:
            print(f"\n{'='*70}\n{text.center(70)}\n{'='*70}")

    @staticmethod
    def table_metrics(fold, epoch, metrics, train_loss):
        """Crea una tabella per visualizzare le performance dell'epoca."""
        if RI_AVAILABLE:
            table = Table(title=f"Epoch {epoch+1} Summary (Fold {fold})", show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="dim")
            table.add_column("Value", justify="right")
            
            table.add_row("Train Loss", f"{train_loss:.4f}")
            table.add_row("Val Loss", f"{metrics['loss']:.4f}")
            table.add_row("F1 (Optimized)", f"[bold green]{metrics['f1']:.4f}[/bold green]")
            table.add_row("Best Threshold", f"{metrics['best_threshold']:.2f}")
            table.add_row("Human F1", f"{metrics['human_f1']:.4f}")
            table.add_row("Machine F1", f"{metrics['machine_f1']:.4f}")
            table.add_row("Precision (M)", f"{metrics['precision_machine']:.4f}")
            table.add_row("Recall (M)", f"{metrics['recall_machine']:.4f}")
            console.print(table)
        else:
            print(f"\n[Ep {epoch+1}] T-Loss: {train_loss:.4f} | V-Loss: {metrics['loss']:.4f} | F1: {metrics['f1']:.4f}")
            print(f"         Thresh: {metrics['best_threshold']:.2d} | H-F1: {metrics['human_f1']:.3f} | M-F1: {metrics['machine_f1']:.3f}")

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch_idx, total_epochs, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    len_dl = len(dataloader)
    
    # Progress bar personalizzata
    pbar = tqdm(dataloader, desc=f"  Training", leave=False, colour="green")
    
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        lang_ids = batch["lang_ids"].to(device)
        
        # Scheduling Alpha DANN
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
        pbar.set_postfix({"L": f"{loss.item():.3f}", "α": f"{alpha:.2f}"})
        
    return running_loss / len_dl

if __name__ == "__main__":
    load_dotenv()
    with open("src/src_TaskA/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    
    # 1. Caricamento e Preparazione
    df_train_raw, df_val_raw = load_data_raw(config)
    full_df = pd.concat([df_train_raw, df_val_raw]).reset_index(drop=True)
    
    language_map = {l: i for i, l in enumerate(sorted(full_df['language'].unique()))}
    config["model"]["num_languages"] = len(language_map)
    
    logger.info(f"Language Map: {language_map}")
    
    # 2. K-Fold Loop
    skf = StratifiedKFold(n_splits=config["training"]["k_folds"], shuffle=True, random_state=42)
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(full_df, full_df['label'])):
        ConsoleUX.header(fold, config["training"]["k_folds"], config["model"]["model_name"])
        
        train_df = balance_dataframe(full_df.iloc[t_idx], config)
        val_df = full_df.iloc[v_idx]
        
        logger.info(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Ratio: {len(train_df)/len(val_df):.2f}")

        train_ds = CodeDataset(train_df, tokenizer, language_map, augment=True)
        val_ds = CodeDataset(val_df, tokenizer, language_map, augment=False)
        
        train_dl = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=config["training"]["batch_size"]*2, shuffle=False, num_workers=4)
        
        dann_weights = get_dann_class_weights(train_df, language_map, device)
        model = CodeClassifier(config, dann_lang_weights=dann_weights).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=0.05)
        
        # Configurazione Scheduler
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
            
            # Valutazione con Threshold Tuning
            metrics = evaluate(model, val_dl, device)
            
            # Logging Tabellare Professionale
            ConsoleUX.table_metrics(fold, epoch, metrics, avg_train_loss)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                patience_counter = 0
                save_path = os.path.join(config["training"]["checkpoint_dir"], f"best_fold{fold}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'threshold': metrics['best_threshold'],
                    'config': config,
                    'f1': best_f1
                }, save_path)
                logger.info(f"[SAVED] Best F1 updated: {best_f1:.4f}\n")
            else:
                patience_counter += 1
                if patience_counter >= config["training"].get("patience", 5):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Cleanup memoria per il prossimo fold
        del model, optimizer, scheduler, train_dl, val_dl
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("\nTraining completato su tutti i fold!")