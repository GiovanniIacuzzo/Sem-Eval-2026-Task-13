import os
import sys
import logging
import yaml
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler 
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix
from comet_ml import Experiment

# Imports locali
from src_TaskB.models.model import CodeClassifier
from src_TaskB.dataset.dataset import load_data
from src_TaskB.utils.utils import evaluate

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
# Ottimizzazioni CUDA
torch.backends.cudnn.benchmark = True
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False

def save_checkpoint(model, path, is_peft=False):
    """
    Salva il modello gestendo correttamente le parti custom (AttentionHead).
    """
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving model to {path}...")
    
    if is_peft:
        # 1. Salva gli adapter LoRA (standard HF)
        model.base_model.save_pretrained(path)
        
        # 2. Salva le componenti Custom che LoRA ignora (Head + Classifier)
        custom_state = {
            'classifier': model.classifier.state_dict(),
            'attention_pooler': model.attention_pooler.state_dict()
        }
        torch.save(custom_state, os.path.join(path, "custom_components.pt"))
    else:
        # Full model save
        torch.save(model.state_dict(), os.path.join(path, "full_model.bin"))

# -----------------------------------------------------------------------------
# Training Routine (Optimized for T4)
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    predictions, references = [], []
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(progress_bar):
        # Spostamento su GPU non-blocking per parallelizzare data loading e compute
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        # Mixed Precision Context
        with autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(input_ids, attention_mask, labels=labels)
            loss = loss / accumulation_steps

        # Scaled Backward Pass (CRITICO PER T4/Float16)
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            # Unscale prima del clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step ottimizzatore e scaler
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        # Metrics
        current_loss = loss.item() * accumulation_steps
        running_loss += current_loss
        
        progress_bar.set_postfix({"Loss": f"{current_loss:.4f}", "LR": f"{scheduler.get_last_lr()[0]:.1e}"})

        # Raccogli predizioni per metriche (detach per staccare dal grafo)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        predictions.extend(preds)
        references.extend(labels_cpu)

    metrics = model.compute_metrics(predictions, references)
    metrics["loss"] = running_loss / len(dataloader)
    
    return metrics

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src_TaskB/config/config.yaml")
    args = parser.parse_args()
    
    ConsoleUX.print_banner(f"SemEval 2026 Task 13 - T4 CUDA Optimized")

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
    if not torch.cuda.is_available():
        logger.warning("CUDA not found! Training will be extremely slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 4. Model Init
    model_wrapper = CodeClassifier(config)
    model_wrapper.to(device)

    is_peft = hasattr(model_wrapper, 'use_lora') and model_wrapper.use_lora

    # 5. Data Loading
    train_dataset, val_dataset, _, _ = load_data(config, model_wrapper.tokenizer)

    # Worker Setup for Linux/CUDA (T4 solitamente ha 2 o 4 CPU cores su Colab/Kaggle)
    num_workers = 4 
    
    train_dl = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,  # CRITICO per velocità GPU
        prefetch_factor=2
    )
    
    val_dl = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    # 6. Optimization Setup
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_wrapper.parameters()), 
        lr=float(config["training"]["learning_rate"]),
        weight_decay=0.01
    )
    
    # GradScaler per AMP (Automatic Mixed Precision)
    scaler = GradScaler()
    
    acc_steps = config["training"].get("gradient_accumulation_steps", 1)
    num_epochs = config["training"].get("num_epochs", 5)
    total_steps = len(train_dl) * num_epochs // acc_steps
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config["training"]["learning_rate"]),
        total_steps=total_steps,
        pct_start=0.1 # 10% warmup
    )

    # 7. Training Loop
    best_f1 = float("-inf")
    patience = config["training"].get("early_stop_patience", 3)
    patience_counter = 0

    checkpoint_dir = config["training"]["checkpoint_dir"]
    best_model_path = os.path.join(checkpoint_dir, "best_model")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_metrics = train_one_epoch(
            model_wrapper, train_dl, optimizer, scheduler, scaler, device, acc_steps
        )
        ConsoleUX.log_metrics("Train", train_metrics)
        
        # Validate
        torch.cuda.empty_cache() # Pulisci cache prima della validazione
        val_metrics, val_preds, val_refs = evaluate(model_wrapper, val_dl, device)
        ConsoleUX.log_metrics("Valid", val_metrics)

        # Logging
        experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
        experiment.log_metrics(val_metrics, prefix="Val", step=epoch)

        # Checkpointing Strategy
        current_f1 = val_metrics["f1"]
        if current_f1 > best_f1:
            logger.info(f" New Best F1: {current_f1:.4f} (was {best_f1:.4f})")
            best_f1 = current_f1
            patience_counter = 0
            
            save_checkpoint(model_wrapper, best_model_path, is_peft=is_peft)
            
            # Confusion Matrix on Comet
            cm = confusion_matrix(val_refs, val_preds)
            experiment.log_confusion_matrix(matrix=cm, title=f"CM_Epoch_{epoch}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    experiment.end()
    logger.info("Training Complete.")