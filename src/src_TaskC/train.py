import os
import sys
import logging
import yaml
import torch
import numpy as np
import argparse
import gc
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from comet_ml import Experiment

from src.src_TaskC.models.model import CodeClassifier
from src.src_TaskC.dataset.dataset import CodeDataset, load_data_for_kfold, get_dynamic_language_map, get_class_weights
from src.src_TaskC.utils.utils import evaluate

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

# -----------------------------------------------------------------------------
# Training Routine
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
        
        del input_ids, attention_mask, labels, lang_ids, logits, loss
    
    metrics = model.compute_metrics(predictions, references)
    metrics["loss"] = running_loss / len_dataloader
    return metrics

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    ConsoleUX.print_banner("SemEval 2026 - Task 13 - subtask C")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/src_TaskC/config/config.yaml") 
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    # --- 1. DATA PATH SETUP ---
    DATA_ROOT = os.getenv("DATA_PATH", "./data")
    task_data_dir = os.path.join(DATA_ROOT, "Task_C")
    
    train_file = os.path.join(task_data_dir, "train.parquet")
    val_file = os.path.join(task_data_dir, "validation.parquet")

    if not os.path.exists(train_file):
        logger.error(f"DATA NOT FOUND! Expected 'train.parquet' in: {task_data_dir}")
        logger.error("Please ensure you have downloaded the data into the './data/Task_C/' folder.")
        sys.exit(1)
    
    logger.info(f"Data Source: {task_data_dir}")
    
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Config not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["data"]["train_path"] = train_file
    config["data"]["val_path"] = val_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # =========================================================================
    # 2. SETUP DATA
    # =========================================================================
    full_df = load_data_for_kfold(config)
    
    language_map = get_dynamic_language_map(full_df)
    config["model"]["num_languages"] = len(language_map)
    
    model_name = config["model"]["model_name"]
    logger.info(f"Loading Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    y_all = full_df['label'].values
    
    kf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME", "semeval-task13-subtaskc"),
        workspace=os.getenv("COMET_WORKSPACE"),
        auto_metric_logging=False
    )
    experiment.log_parameters(config)

    # =========================================================================
    # 3. K-FOLD LOOP
    # =========================================================================
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(y_all)), y_all)):
        ConsoleUX.print_banner(f"FOLD {fold+1}/{args.k_folds}")
        
        train_df_fold = full_df.iloc[train_idx]
        val_df_fold = full_df.iloc[val_idx]
        
        class_weights = get_class_weights(train_df_fold, device)
        
        logger.info(f"Creating Datasets for Fold {fold+1}...")
        train_dataset = CodeDataset(
            dataframe=train_df_fold, 
            tokenizer=tokenizer, 
            language_map=language_map,
            max_length=config["data"]["max_length"], 
            augment=True
        )
        val_dataset = CodeDataset(
            dataframe=val_df_fold, 
            tokenizer=tokenizer, 
            language_map=language_map,
            max_length=config["data"]["max_length"], 
            augment=False
        )
        logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

        model = CodeClassifier(config, class_weights=class_weights)
        
        if model.num_languages != len(language_map):
            logger.info(f"Adjusting DANN head to {len(language_map)} languages.")
            import torch.nn as nn
            model.language_classifier = nn.Sequential(
                nn.Linear(model.hidden_size, model.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(model.hidden_size // 2, len(language_map))
            )
            model._init_weights(model.language_classifier)
            
        model.to(device)
        
        num_workers = 4
        train_dl = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], 
                             shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        val_dl = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], 
                           shuffle=False, num_workers=num_workers, pin_memory=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=0.01)
        scaler = GradScaler()
        
        acc_steps = config["training"].get("gradient_accumulation_steps", 1)
        num_epochs = config["training"].get("num_epochs", 5) 
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
            
            val_metrics, val_preds, val_refs = evaluate(model, val_dl, device, verbose=False)
            
            ConsoleUX.log_metrics(f"F{fold+1}-Ep{epoch+1}", val_metrics)
            
            experiment.log_metrics(train_metrics, prefix=f"Fold{fold}_Train", step=epoch)
            experiment.log_metrics(val_metrics, prefix=f"Fold{fold}_Val", step=epoch)

            if val_metrics["f1"] > best_fold_f1:
                logger.info(f"New Best F1 for Fold {fold+1}: {val_metrics['f1']:.4f}")
                best_fold_f1 = val_metrics["f1"]
                patience_counter = 0
                
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

        del model, optimizer, scheduler, scaler, train_dl, val_dl, train_dataset, val_dataset
        torch.cuda.empty_cache()
        gc.collect()

    experiment.end()
    logger.info("Training Finished. All folds processed.")