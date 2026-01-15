import os
import sys
import yaml
import torch
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
from dotenv import load_dotenv
from comet_ml import Experiment
from sklearn.metrics import confusion_matrix
from pytorch_metric_learning import losses

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None

from src.src_TaskA.models.model import HybridClassifier
from src.src_TaskA.dataset.dataset import load_data
from src.src_TaskB.utils.utils import set_seed, evaluate_model

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

class ConsoleUX:
    """Gestisce l'output formattato per monitorare il training."""
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        log_str = f"[{stage}] "
        priority_keys = ["loss", "f1_macro", "acc", "task_loss", "supcon_loss"]
        
        for k in priority_keys:
            if k in metrics:
                log_str += f"{k}: {metrics[k]:.4f} | "
        
        for k, v in metrics.items():
            if k not in priority_keys:
                if isinstance(v, float):
                    log_str += f"{k}: {v:.4f} | "
        
        logger.info(log_str.strip(" | "))

def save_checkpoint(model, tokenizer, path, epoch, metrics, config):
    """Salva stato modello, tokenizer e configurazione."""
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving checkpoint to {path}...")
    
    tokenizer.save_pretrained(path)
    
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(path, "model_state.bin"))
    
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.dump(config, f)
        
    with open(os.path.join(path, "training_meta.yaml"), "w") as f:
        yaml.dump({"epoch": epoch, "metrics": metrics}, f)

# -----------------------------------------------------------------------------
# 2. TRAINING ENGINE
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, 
                   epoch_idx, acc_steps=1, supcon_fn=None):
    model.train()
    
    tracker = {"loss": 0.0, "task_loss": 0.0, "supcon_loss": 0.0}
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(dataloader, desc=f"Train Ep {epoch_idx+1}", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        feats = batch["extra_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            _, task_loss, combined_features = model(
                input_ids, attention_mask, feats, labels=labels
            )
            
            supcon_loss = torch.tensor(0.0, device=device)
            if supcon_fn is not None:
                features_norm = torch.nn.functional.normalize(combined_features, dim=1)
                supcon_loss = supcon_fn(features_norm, labels)
            
            total_loss = (task_loss + 0.1 * supcon_loss) / acc_steps

        scaler.scale(total_loss).backward()

        if (step + 1) % acc_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if scheduler is not None:
                scheduler.step()
        
        current_loss = total_loss.item() * acc_steps
        tracker["loss"] += current_loss
        tracker["task_loss"] += task_loss.item()
        tracker["supcon_loss"] += supcon_loss.item()
        
        pbar.set_postfix({
            "Loss": f"{current_loss:.3f}",
            "SupCon": f"{supcon_loss.item():.3f}" if supcon_fn else "0.0"
        })

    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in tracker.items()}

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="SemEval Task A - Training")
    parser.add_argument("--config", type=str, default="src/src_TaskA/config/config.yaml")
    args = parser.parse_args()
    
    ConsoleUX.print_banner("SemEval Task 13 - Subtask A")

    # 1. Configurazione
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)
        
    config = raw_config
    common_cfg = config.get("common", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    set_seed(common_cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # 2. Comet ML Tracking
    api_key = os.getenv("COMET_API_KEY")
    experiment = None
    if api_key:
        experiment = Experiment(
            api_key=api_key,
            project_name=common_cfg.get("project_name", "semeval-task-a"),
            auto_metric_logging=False
        )
        experiment.log_parameters(config)
        experiment.add_tag("TaskA_Hybrid")

    # 3. Data Loading
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    
    logger.info("Initializing Datasets...")
    train_ds, val_ds = load_data(config, tokenizer)
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=train_cfg["batch_size"], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=train_cfg["batch_size"], 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # 4. Model Setup
    model = HybridClassifier(config)
    model.to(device)
    
    # 5. Optimizer & Scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=float(train_cfg["learning_rate"]), 
        weight_decay=train_cfg.get("weight_decay", 0.01)
    )
    
    supcon_fn = None
    if train_cfg.get("use_supcon", False):
        logger.info("Activating Supervised Contrastive Loss...")
        supcon_fn = losses.SupConLoss(temperature=0.1).to(device)

    scaler = GradScaler()
    acc_steps = train_cfg.get("gradient_accumulation_steps", 1)
    total_steps = (len(train_dl) // acc_steps) * train_cfg["num_epochs"]
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=float(train_cfg["learning_rate"]), 
        total_steps=total_steps,
        pct_start=0.1
    )

    # 6. Training Loop
    best_f1 = 0.0
    patience = train_cfg.get("early_stop_patience", 3)
    patience_counter = 0
    checkpoint_dir = train_cfg["checkpoint_dir"]
    
    label_names = ["Human", "AI"]

    logger.info(f"Starting Training for {train_cfg['num_epochs']} epochs...")

    for epoch in range(train_cfg["num_epochs"]):
        ConsoleUX.print_banner(f"Epoch {epoch+1}/{train_cfg['num_epochs']}")
        
        # --- TRAIN ---
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, 
            epoch, acc_steps, supcon_fn
        )
        ConsoleUX.log_metrics("Train", train_metrics)
        
        if experiment:
            experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
            experiment.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

        # --- VALIDATION ---
        val_metrics, val_preds, val_labels, report = evaluate_model(model, val_dl, device, label_names)
        
        ConsoleUX.log_metrics("Val", val_metrics)
        logger.info(f"\n{report}")
        
        if experiment:
            experiment.log_metrics(val_metrics, prefix="Val", step=epoch)

        # --- CHECKPOINTING ---
        current_f1 = val_metrics["f1_macro"]
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            
            logger.info(f"--> New Best F1: {best_f1:.4f}. Saving Model...")
            
            # Salvataggio
            save_checkpoint(
                model, tokenizer, 
                os.path.join(checkpoint_dir, "best_model"), 
                epoch, val_metrics, config
            )
            
            if experiment:
                cm = confusion_matrix(val_labels, val_preds)
                experiment.log_confusion_matrix(matrix=cm, labels=label_names, title="Best Model CM")
                
        else:
            patience_counter += 1
            logger.warning(f"--> No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            ConsoleUX.print_banner("EARLY STOPPING TRIGGERED")
            break

    if experiment:
        experiment.end()
    
    logger.info("Training Finished.")