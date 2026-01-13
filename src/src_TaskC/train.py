import os
import sys
import logging
import yaml
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer
from comet_ml import Experiment
from pytorch_metric_learning import losses

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from src.src_TaskC.models.model import CodeClassifier
from src.src_TaskC.dataset.dataset import CodeDataset, load_data_for_training, get_class_weights
from src.src_TaskC.utils.utils import evaluate, set_seed

# -----------------------------------------------------------------------------
# Logger & UX Setup
# -----------------------------------------------------------------------------
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
        keys = sorted(metrics.keys(), key=lambda x: (0 if 'f1' in x else 1 if 'acc' in x else 2, x))
        log_str = f"[{stage}] "
        for k in keys:
            v = metrics[k]
            if "class" in k: 
                short_k = k.replace('f1_class_', 'C')
                log_str += f"{short_k}: {v:.3f} | "
            elif isinstance(v, float):
                log_str += f"{k}: {v:.4f} | "
            else:
                log_str += f"{k}: {v} | "
        logger.info(log_str.strip(" | "))

# -----------------------------------------------------------------------------
# Training Routine
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, 
                   epoch_idx, accumulation_steps=1, contrastive_fn=None):
    
    model.train()
    
    running_metrics = {"loss": 0.0, "task_loss": 0.0, "con_loss": 0.0}
    correct_preds = 0
    total_preds = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    progress_bar = tqdm(dataloader, desc=f"Ep {epoch_idx+1} Train", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        extra_features = batch["extra_features"].to(device, non_blocking=True)

        with autocast(device_type='cuda', dtype=torch.float16):
            logits, loss_task, proj_features = model(
                input_ids, attention_mask, 
                labels=labels, 
                extra_features=extra_features
            )
            
            loss_con = torch.tensor(0.0, device=device)
            if contrastive_fn is not None:
                loss_con = contrastive_fn(proj_features, labels)

            total_loss = (loss_task + 0.5 * loss_con) / accumulation_steps

        scaler.scale(total_loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if scheduler is not None:
                scheduler.step()
        
        current_loss = total_loss.item() * accumulation_steps
        running_metrics["loss"] += current_loss
        running_metrics["task_loss"] += loss_task.item()
        running_metrics["con_loss"] += loss_con.item()
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
        
        progress_bar.set_postfix({
            "L": f"{current_loss:.3f}", 
            "T": f"{loss_task.item():.3f}",
            "C": f"{loss_con.item():.3f}",
            "Acc": f"{correct_preds/total_preds:.2f}"
        })

    avg_loss = running_metrics["loss"] / len(dataloader)
    epoch_acc = correct_preds / total_preds
    
    return {
        "loss": avg_loss, 
        "accuracy": epoch_acc,
        "task_loss": running_metrics["task_loss"] / len(dataloader),
        "con_loss": running_metrics["con_loss"] / len(dataloader)
    }

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    ConsoleUX.print_banner("SemEval 2026 - Task 13 - Subtask C")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/src_TaskC/config/config.yaml") 
    args = parser.parse_args()

    # --- 1. SETUP CONFIG & DEVICE ---
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # =========================================================================
    # 2. DATA LOADING
    # =========================================================================
    train_df, val_df = load_data_for_training(config)
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    
    raw_weights = get_class_weights(train_df, device)
    if raw_weights is not None:
        logger.info(f"Class Weights (Focal Loss): {raw_weights.cpu().numpy()}")
    else:
        logger.warning("Using uniform class weights.")
        
    # Dataset
    train_dataset = CodeDataset(train_df, tokenizer, max_length=config["data"]["max_length"], is_train=True)
    val_dataset = CodeDataset(val_df, tokenizer, max_length=config["data"]["max_length"], is_train=False)

    # Dataloaders
    train_dl = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=config.get("num_workers", 4), 
        pin_memory=True, 
        drop_last=True
    )
    
    val_dl = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=config.get("num_workers", 4), 
        pin_memory=True
    )

    # =========================================================================
    # 3. MODEL & EXPERIMENT SETUP
    # =========================================================================
    model = CodeClassifier(config, class_weights=raw_weights)
    model.to(device)

    # Comet ML Logging
    api_key = os.getenv("COMET_API_KEY")
    experiment = None
    if api_key:
        experiment = Experiment(
            api_key=api_key,
            project_name=os.getenv("COMET_PROJECT_NAME", "semeval-task13-subtaskc"),
            workspace=os.getenv("COMET_WORKSPACE"),
            auto_metric_logging=False
        )
        experiment.log_parameters(config)
        experiment.add_tag("Fusion_SupCon")

    # Optimizer & Scheduler
    lr = float(config["training"]["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=config["training"].get("weight_decay", 0.01)
    )
    
    scaler = GradScaler()
    
    acc_steps = config["training"].get("gradient_accumulation_steps", 1)
    num_epochs = config["training"].get("num_epochs", 5) 
    total_steps = (len(train_dl) // acc_steps) * num_epochs
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        total_steps=total_steps, 
        pct_start=0.1
    )

    # Contrastive Loss (SupCon)
    logger.info("Initializing SupConLoss...")
    contrastive_fn = losses.SupConLoss(temperature=0.1).to(device)

    # =========================================================================
    # 4. TRAINING LOOP
    # =========================================================================
    best_f1 = 0.0
    patience = config["training"].get("early_stop_patience", 4)
    patience_counter = 0
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        # --- TRAIN ---
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, 
            epoch, accumulation_steps=acc_steps, contrastive_fn=contrastive_fn
        )
        
        # --- VALIDATION ---
        val_metrics, _, _ = evaluate(model, val_dl, device, verbose=False)
        
        # --- LOGGING ---
        ConsoleUX.log_metrics(f"Ep{epoch+1}", val_metrics)
        if experiment:
            experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
            experiment.log_metrics(val_metrics, prefix="Val", step=epoch)
            experiment.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

        # --- CHECKPOINTING ---
        current_f1 = val_metrics.get("f1_macro", val_metrics.get("f1", 0.0))
        
        if current_f1 > best_f1:
            logger.info(f"---> New Best F1: {current_f1:.4f} (prev: {best_f1:.4f})")
            best_f1 = current_f1
            patience_counter = 0
            
            save_path = os.path.join(checkpoint_dir, "best_model.bin")
            torch.save(model.state_dict(), save_path)
            
            with open(os.path.join(checkpoint_dir, "training_meta.yaml"), "w") as f:
                yaml.dump({"best_epoch": epoch, "best_f1": best_f1}, f)
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            ConsoleUX.print_banner("EARLY STOPPING TRIGGERED")
            break

    if experiment:
        experiment.end()
    
    logger.info("Training Finished Successfully.")