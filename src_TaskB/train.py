import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import sys
import logging
import yaml
import torch
import argparse
import gc
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

from comet_ml import Experiment

# Assicurati che i path siano corretti rispetto alla tua struttura
from src_TaskB.models.model import CodeClassifier
from src_TaskB.dataset.dataset import load_data
from src_TaskB.utils.utils import evaluate

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
# Evita deadlock su Mac/Linux con i tokenizer paralleli
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConsoleUX:
    """Helper class for cleaner console output."""
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        """Formats metrics into a readable string."""
        log_str = f"[{stage}] "
        for k, v in metrics.items():
            if isinstance(v, float):
                log_str += f"{k}: {v:.4f} | "
            else:
                log_str += f"{k}: {v} | "
        logger.info(log_str.strip(" | "))

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    """Reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def clear_memory(device):
    """Aggressive memory cleanup for M2."""
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

def save_checkpoint(model, path, is_peft=False):
    """Saves model efficiently. Supports LoRA/PEFT."""
    logger.info(f"Saving model to {path}...")
    if is_peft:
        # Se è un modello PEFT, salviamo solo gli adapter per risparmiare spazio
        model.base_model.save_pretrained(path)
        # Salviamo anche la head classifier separatamente poiché non fa parte di PEFT base
        torch.save(model.classifier.state_dict(), os.path.join(path, "classifier_head.pt"))
    else:
        # Standard full model save
        torch.save(model.state_dict(), path)

# -----------------------------------------------------------------------------
# Training Routine
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=1):
    """
    Executes one training epoch with Mixed Precision and Gradient Accumulation.
    """
    model.train()
    running_loss = 0.0
    predictions, references = [], []
    
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True)
    
    # Mixed Precision Setup
    device_type = device.type if device.type in ['cuda', 'mps'] else 'cpu'
    dtype = torch.float16 if device_type != 'cpu' else torch.float32
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        with autocast(device_type=device_type, dtype=dtype):
            logits, loss = model(input_ids, attention_mask, labels=labels)
            # Normalize loss for accumulation
            loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        # Metrics & Logging
        current_loss = loss.item() * accumulation_steps
        running_loss += current_loss
        
        progress_bar.set_postfix({"Loss": f"{current_loss:.4f}", "LR": f"{scheduler.get_last_lr()[0]:.1e}"})

        # Save metrics only periodically to save CPU cycles if dataset is huge
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        predictions.extend(preds)
        references.extend(labels_cpu)
        
        # Explicit delete to help MPS allocator
        del input_ids, attention_mask, labels, logits, loss

    metrics = model.compute_metrics(predictions, references)
    metrics["loss"] = running_loss / len(dataloader)
    
    return metrics

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    set_seed(42)
    
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src_TaskB/config/config.yaml")
    args = parser.parse_args()
    
    ConsoleUX.print_banner(f"SemEval 2026 Task 13 - M2 Optimized Training")

    # 1. Load Config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # 2. Experiment Tracking
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME"),
        workspace=os.getenv("COMET_WORKSPACE"),
        auto_metric_logging=False
    )
    experiment.log_parameters(config)

    # 3. Device Setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Compute Device: {device}")
    
    # 4. Model Init
    logger.info(f"Initializing Model: {config['model']['model_name']}...")
    model_wrapper = CodeClassifier(config)
    model_wrapper.to(device)

    # Detect LoRA usage for saving strategy
    is_peft = hasattr(model_wrapper, 'use_lora') and model_wrapper.use_lora

    # 5. Data Loading
    logger.info("Loading Datasets...")
    train_dataset, val_dataset, _, _ = load_data(config, model_wrapper.tokenizer)

    # Worker Setup for M2
    # MPS/Mac works best with 0 (main thread) or 1-2 workers. Too many = overhead.
    num_workers = 2 
    
    train_dl = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, # Important for sliding window chunks
        num_workers=num_workers, 
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_dl = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=False
    )

    # 6. Optimization
    # Filters parameters that require gradients (handles LoRA automatically)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_wrapper.parameters()), 
        lr=float(config["training"]["learning_rate"]),
        weight_decay=0.01
    )
    
    acc_steps = config["training"].get("gradient_accumulation_steps", 1)
    num_epochs = config["training"].get("num_epochs", 5)
    total_steps = len(train_dl) * num_epochs // acc_steps
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config["training"]["learning_rate"]),
        total_steps=total_steps,
        pct_start=0.1
    )

    # 7. Training Loop
    best_metric = float("-inf") # Assuming Higher F1 is better
    eval_metric = "f1"
    patience = config["training"].get("early_stop_patience", 3)
    patience_counter = 0

    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model")

    ConsoleUX.print_banner(f"Starting Training ({num_epochs} Epochs)")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_metrics = train_one_epoch(model_wrapper, train_dl, optimizer, scheduler, device, acc_steps)
        ConsoleUX.log_metrics("Train", train_metrics)
        
        # Validate
        # Free memory before validation
        clear_memory(device)
        
        val_metrics, val_preds, val_refs = evaluate(model_wrapper, val_dl, device)
        ConsoleUX.log_metrics("Valid", val_metrics)

        # Logging
        experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
        experiment.log_metrics(val_metrics, prefix="Val", step=epoch)

        # Save Best Model
        current_score = val_metrics[eval_metric]
        if current_score > best_metric:
            logger.info(f"⭐ New Best {eval_metric}: {current_score:.4f} (was {best_metric:.4f})")
            best_metric = current_score
            patience_counter = 0
            
            # Save IMMEDIATELY to disk instead of holding in RAM
            save_checkpoint(model_wrapper, best_model_path, is_peft=is_peft)
            
            # Save confusion matrix
            cm = confusion_matrix(val_refs, val_preds)
            experiment.log_confusion_matrix(matrix=cm, title=f"CM_Epoch_{epoch}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
            
        clear_memory(device)

    # 8. Final Test
    experiment.end()
    logger.info("Training Complete.")