import os
import sys
import yaml
import torch
import argparse
import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler 
from transformers import AutoTokenizer
from dotenv import load_dotenv
from comet_ml import Experiment

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None

from src.src_TaskA.models.model import CodeClassifier
from src.src_TaskA.dataset.dataset import load_data_lolo 
from src.src_TaskA.utils.utils import evaluate_model, DynamicCollate

# -----------------------------------------------------------------------------
# 1. SETUP & UTILS
# -----------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Garantisce la riproducibilità totale."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ConsoleUX:
    """Utility per output console puliti e professionali."""
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        log_str = f"[{stage}] "
        # Aggiungiamo chiavi per monitorare l'avversario se presenti
        keys = ["loss", "loss_task", "loss_adv", "accuracy", "f1_macro"]
        for k in keys:
            if k in metrics:
                log_str += f"{k}: {metrics[k]:.4f} | "
        logger.info(log_str.strip(" | "))

def save_checkpoint(model, tokenizer, path, epoch, metrics, config):
    """Salva modello, tokenizer e metadati completi."""
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving checkpoint to {path}...")
    
    torch.save(model.state_dict(), os.path.join(path, "model_state.bin"))
    tokenizer.save_pretrained(path)
    
    with open(os.path.join(path, "training_meta.yaml"), "w") as f:
        yaml.dump({"epoch": epoch, "metrics": metrics, "config": config}, f)

# -----------------------------------------------------------------------------
# 2. TRAINING ENGINE (ADVERSARIAL AWARE)
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch_idx):
    model.train()

    running_loss = 0.0
    running_loss_task = 0.0
    running_loss_adv = 0.0
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch_idx+1}", leave=False, dynamic_ncols=True)
    
    for batch in pbar:
        # Spostamento su GPU
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        # Gestione Extra Features (Feature Purificate)
        extra_features = batch.get("extra_features", None)
        if extra_features is not None:
            extra_features = extra_features.to(device, non_blocking=True)
            
        # Gestione Adversarial Labels (Language IDs)
        lang_labels = batch.get("language_labels", None)
        if lang_labels is not None:
            lang_labels = lang_labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision Forward
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(
                input_ids, 
                attention_mask, 
                labels=labels, 
                extra_features=extra_features,
                language_labels=lang_labels
            )
            loss = outputs["loss"]
            detailed_losses = outputs.get("detailed_losses", {})

        # Backward scalato
        scaler.scale(loss).backward()
        
        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        # Logging Accumulators
        current_loss = loss.item()
        running_loss += current_loss
        running_loss_task += detailed_losses.get("loss_task", 0.0)
        running_loss_adv += detailed_losses.get("loss_adv", 0.0)
        
        pbar.set_postfix({
            "Tot": f"{current_loss:.3f}", 
            "Task": f"{detailed_losses.get('loss_task', 0):.3f}",
            "Adv": f"{detailed_losses.get('loss_adv', 0):.3f}"
        })

    # Calcolo Medie
    n_batches = len(dataloader)
    return {
        "loss": running_loss / n_batches,
        "loss_task": running_loss_task / n_batches,
        "loss_adv": running_loss_adv / n_batches
    }

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="SemEval Task A - Domain Adversarial Training")
    parser.add_argument("--config", type=str, default="src/src_TaskA/config/config.yaml", help="Path to config file")    
    parser.add_argument("--holdout_language", type=str, default=None, 
                        help="Language to exclude from training (for OOD Validation). E.g., 'Python'")
    
    args = parser.parse_args()
    
    ConsoleUX.print_banner("SemEval 2026 - Task 13 - Subtask A (DANN Mode)")
    
    # 1. Load Config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found at {args.config}")
        sys.exit(1)
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)["common"]
        
    set_seed(config["seed"])
    
    # 2. Setup Comet ML
    api_key = os.getenv("COMET_API_KEY")
    if api_key:
        experiment = Experiment(
            api_key=api_key,
            project_name=config.get("project_name", "semeval-task-a-ood"),
            auto_metric_logging=False
        )
        experiment.set_name(f"OOD-{args.holdout_language}" if args.holdout_language else "Standard-Run")
        experiment.log_parameters(config)
        if args.holdout_language:
            experiment.log_parameter("holdout_language", args.holdout_language)
    else:
        experiment = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")

    # 3. Tokenizer
    logger.info(f"Loading Tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # 4. Data Loading (LOLO Strategy)
    train_dataset, val_dataset, lang2id = load_data_lolo(
        config, 
        tokenizer, 
        holdout_language=args.holdout_language
    )
    
    config["num_languages"] = len(lang2id)
    logger.info(f"Model configured for {config['num_languages']} adversarial languages.")
    
    collate_fn = DynamicCollate(tokenizer)
    
    train_dl = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config["num_workers"],
        pin_memory=True, 
        persistent_workers=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_dl = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"] * 2, 
        shuffle=False, 
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 5. Model Init
    model = CodeClassifier(config)
    model.to(device)
    
    # 6. Optimizer & Scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=float(config["learning_rate"]), 
        weight_decay=0.01
    )
    
    scaler = GradScaler()
    
    total_steps = len(train_dl) * config["num_epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=float(config["learning_rate"]), 
        total_steps=total_steps,
        pct_start=0.1
    )

    # 7. Training Loop
    best_f1 = float("-inf")
    patience_counter = 0
    patience = config.get("early_stop_patience", 3)
    checkpoint_dir = config["checkpoint_dir"]
    
    # Se stiamo facendo LOLO, salviamo in una cartella specifica
    if args.holdout_language:
        checkpoint_dir = os.path.join(checkpoint_dir, f"holdout_{args.holdout_language}")

    logger.info("Starting Training...")
    
    for epoch in range(config["num_epochs"]):
        ConsoleUX.print_banner(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # --- TRAIN ---
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, epoch
        )
        ConsoleUX.log_metrics("Train", train_metrics)
        
        if experiment:
            experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
            experiment.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

        # --- VALIDATION ---
        val_metrics, val_report = evaluate_model(model, val_dl, device)
        
        ConsoleUX.log_metrics("Valid", val_metrics)
        logger.info(f"\n{val_report}")
        
        if experiment:
            experiment.log_metrics(val_metrics, prefix="Val", step=epoch)

        # --- CHECKPOINTING ---
        current_f1 = val_metrics["f1_macro"]
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            
            save_path = os.path.join(checkpoint_dir, "best_model")
            save_checkpoint(model, tokenizer, save_path, epoch, val_metrics, config)
            
            logger.info(f"---> New Best F1: {best_f1:.4f}. Model Saved.")
            
            if experiment:
                experiment.log_metric("best_f1", best_f1, step=epoch)
        else:
            patience_counter += 1
            logger.warning(f"No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            ConsoleUX.print_banner("EARLY STOPPING TRIGGERED")
            break
    
    if experiment:
        experiment.end()
    
    logger.info("Training Completed Successfully.")