import os
import sys
import yaml
import torch
import argparse
import logging
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
from src.src_TaskA.dataset.dataset import load_data
from src.src_TaskA.utils.utils import evaluate_model

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
        keys = ["loss", "accuracy", "f1_macro", "precision_macro", "recall_macro"]
        for k in keys:
            if k in metrics:
                log_str += f"{k}: {metrics[k]:.4f} | "
        logger.info(log_str.strip(" | "))

class DynamicCollate:
    """
    Gestisce il padding dinamico. Padda il batch alla lunghezza della sequenza 
    più lunga NEL BATCH, non alla max_length globale. Ottimizza la VRAM.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        
        # Padding intelligente usando il tokenizer
        padded_inputs = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True, 
            return_tensors="pt"
        )

        return {
            "input_ids": padded_inputs["input_ids"],
            "attention_mask": padded_inputs["attention_mask"],
            "labels": labels
        }

def save_checkpoint(model, tokenizer, path, epoch, metrics):
    """Salva modello, tokenizer e metadati."""
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving checkpoint to {path}...")
    
    # Salviamo lo state_dict del modello
    torch.save(model.state_dict(), os.path.join(path, "model_state.bin"))
    tokenizer.save_pretrained(path)
    
    # Salviamo i metadati
    with open(os.path.join(path, "training_meta.yaml"), "w") as f:
        yaml.dump({"epoch": epoch, "metrics": metrics}, f)

# -----------------------------------------------------------------------------
# 2. TRAINING ENGINE
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch_idx):
    model.train()
    running_loss = 0.0
    
    # Barra di progresso dinamica
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch_idx+1}", leave=False, dynamic_ncols=True)
    
    for batch in pbar:
        # Spostamento su GPU
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision Forward
        with autocast(device_type='cuda', dtype=torch.float16):
            # Il modello calcola internamente sia CE che SupCon loss
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs["loss"]

        # Backward scalato
        scaler.scale(loss).backward()
        
        # Gradient Clipping e Optimizer Step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        # Logging
        current_loss = loss.item()
        running_loss += current_loss
        pbar.set_postfix({"Loss": f"{current_loss:.4f}"})

    avg_loss = running_loss / len(dataloader)
    return {"loss": avg_loss}

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="SemEval Task A Training Script")
    parser.add_argument("--config", type=str, default="src/src_TaskA/config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    ConsoleUX.print_banner("SemEval 2026 - Task 13 - Subtask A")
    
    # Load Config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found at {args.config}")
        sys.exit(1)
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)["common"]
        
    set_seed(config["seed"])
    
    # Setup Comet ML
    api_key = os.getenv("COMET_API_KEY")
    if api_key:
        experiment = Experiment(
            api_key=api_key,
            project_name=config.get("project_name", "semeval-task-a"),
            auto_metric_logging=False
        )
        experiment.set_name(config.get("experiment_name", "Run"))
        experiment.log_parameters(config)
        logger.info("Comet ML initialized successfully.")
    else:
        logger.warning("COMET_API_KEY not found. Logging will be local only.")
        experiment = None

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")

    # 1. Tokenizer
    logger.info(f"Loading Tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # 2. Data Loading
    logger.info("Loading Datasets...")
    # La logica di bilanciamento (Python Downsampling) è dentro load_data -> CodeDataset
    train_dataset, val_dataset, _ = load_data(config, tokenizer)
    
    collate_fn = DynamicCollate(tokenizer)
    
    # DataLoader Ottimizzati
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
        # Possiamo raddoppiare il batch in val perché non salviamo i gradienti
        batch_size=config["batch_size"] * 2, 
        shuffle=False, 
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 3. Model Init
    model = CodeClassifier(config)
    model.to(device)
    
    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=float(config["learning_rate"]), 
        weight_decay=0.01
    )
    
    scaler = GradScaler()
    
    # Usiamo OneCycleLR come nel tuo Task B: convergenza più veloce
    total_steps = len(train_dl) * config["num_epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=float(config["learning_rate"]), 
        total_steps=total_steps,
        pct_start=0.1 # 10% warmup
    )

    # 5. Training Loop
    best_f1 = float("-inf")
    patience_counter = 0
    patience = config.get("early_stop_patience", 3)
    checkpoint_dir = config["checkpoint_dir"]

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
            # Loggare il Learning Rate aiuta a debuggare
            experiment.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

        # --- VALIDATION ---
        val_metrics, val_report = evaluate_model(model, val_dl, device)
        
        ConsoleUX.log_metrics("Valid", val_metrics)
        if experiment:
            experiment.log_metrics(val_metrics, prefix="Val", step=epoch)
            
        logger.info(f"\n{val_report}")

        # --- CHECKPOINTING ---
        current_f1 = val_metrics["f1_macro"]
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            
            save_path = os.path.join(checkpoint_dir, "best_model")
            save_checkpoint(model, tokenizer, save_path, epoch, val_metrics)
            
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