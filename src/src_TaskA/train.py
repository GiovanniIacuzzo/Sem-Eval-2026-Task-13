import os
import sys
import logging
import yaml
import torch
import numpy as np
import random
import torch.nn as nn 
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler 
from torch.optim.swa_utils import AveragedModel, SWALR
from dotenv import load_dotenv

from comet_ml import Experiment
from pytorch_metric_learning import losses as metric_losses

# Assumiamo che tu lanci il codice come modulo (python -m src.src_TaskA.train)
from src.src_TaskA.models.model import FusionCodeClassifier 
from src.src_TaskA.dataset.dataset import load_data
from src.src_TaskA.utils.utils import evaluate

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
        log_str = f"[{stage}] "
        for k, v in metrics.items():
            if isinstance(v, float):
                log_str += f"{k}: {v:.4f} | "
            else:
                log_str += f"{k}: {v} | "
        logger.info(log_str.strip(" | "))

# -----------------------------------------------------------------------------
# OPTIMIZER UTILS: LLRD
# -----------------------------------------------------------------------------
def get_llrd_optimizer_params(model, base_lr, weight_decay=0.01, decay_factor=0.95):
    opt_parameters = []
    named_parameters = list(model.named_parameters())
    
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    head_params_names = [n for n, p in named_parameters if "base_model" not in n]
    
    head_params_decay = [p for n, p in named_parameters if n in head_params_names and not any(nd in n for nd in no_decay)]
    head_params_nodecay = [p for n, p in named_parameters if n in head_params_names and any(nd in n for nd in no_decay)]
    
    opt_parameters.append({"params": head_params_decay, "lr": base_lr, "weight_decay": weight_decay})
    opt_parameters.append({"params": head_params_nodecay, "lr": base_lr, "weight_decay": 0.0})
    
    if hasattr(model, "base_model") and hasattr(model.base_model, "encoder"):
        layers = list(model.base_model.encoder.layer)
        layers.reverse()
        lr = base_lr
        
        for layer in layers:
            lr *= decay_factor
            decay = []
            nodecay = []
            for n, p in layer.named_parameters():
                if any(nd in n for nd in no_decay):
                    nodecay.append(p)
                else:
                    decay.append(p)
            opt_parameters.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
            opt_parameters.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})
            
        embeddings_params = list(model.base_model.embeddings.parameters())
        lr *= decay_factor
        opt_parameters.append({"params": embeddings_params, "lr": lr, "weight_decay": weight_decay})
    else:
        backbone_params = [p for n, p in named_parameters if "base_model" in n]
        opt_parameters.append({"params": backbone_params, "lr": base_lr * 0.8, "weight_decay": weight_decay})

    return opt_parameters

# -----------------------------------------------------------------------------
# Training Routine (SIMPLIFIED FOR DEBUGGING)
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, 
                   epoch_idx, total_epochs, accumulation_steps=1, swa_model=None, swa_scheduler=None):
    
    model.train()
    running_loss = 0.0
    
    # --- Inizializzazione Loss Functions ---
    ce_criterion = nn.CrossEntropyLoss()
    
    # COMMENTATO PER DEBUG: Disabilitiamo SupCon e MSE per ora
    # supcon_criterion = metric_losses.SupConLoss(temperature=0.1).to(device)
    # mse_criterion = nn.MSELoss()
    
    optimizer.zero_grad()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        stylo_feats = batch.get("stylo_feats").to(device, non_blocking=True)
        
        # Mixed Precision Context
        with autocast(device_type='cuda', dtype=torch.float16):
            # --- Forward Pass ---
            # Scompattiamo, ma ignoriamo le feature per SupCon/Consistency per ora
            logits, _, _, _ = model(
                input_ids, attention_mask, stylo_feats, labels=None, return_embedding=True
            )
            
            # --- A. Cross Entropy Loss (ONLY THIS ONE) ---
            loss = ce_criterion(logits, labels)
            
            # --- DEBUG: Disabilitiamo le auxiliary losses per evitare il collapse ---
            # loss_supcon = supcon_criterion(supcon_feats, labels)
            # total_loss = loss_ce + (0.5 * loss_supcon)
            # if "has_aug" in batch...
            # loss = total_loss / accumulation_steps
            
            # Scaliamo la loss per accumulation
            loss = loss / accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if swa_model is not None and swa_scheduler is not None:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            elif scheduler is not None:
                scheduler.step()
        
        current_loss = loss.item() * accumulation_steps
        running_loss += current_loss
        
        progress_bar.set_postfix({
            "Loss": f"{current_loss:.4f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.1e}"
        })
        
        del input_ids, attention_mask, labels, logits, loss

    return {"loss": running_loss / len(dataloader)}
     
# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    set_seed(42)
    load_dotenv()
    ConsoleUX.print_banner("SemEval 2026 - Task 13 - subtask A")

    DATA_ROOT = os.getenv("DATA_PATH", "./data")
    task_data_dir = os.path.join(DATA_ROOT, "Task_A")
    
    # Configurazione
    config_path = "src/src_TaskA/config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Autodiscovery paths
    train_files = [f for f in os.listdir(task_data_dir) if "train" in f and f.endswith(".parquet")]
    val_files = [f for f in os.listdir(task_data_dir) if ("val" in f or "dev" in f) and f.endswith(".parquet")]

    if train_files: config["data"]["train_path"] = os.path.join(task_data_dir, train_files[0])
    if val_files: config["data"]["val_path"] = os.path.join(task_data_dir, val_files[0])

    # Experiment Tracking
    api_key = os.getenv("COMET_API_KEY")
    if api_key:
        experiment = Experiment(api_key=api_key, project_name=os.getenv("COMET_PROJECT_NAME"))
        experiment.log_parameters(config)
    else:
        experiment = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # --- MODEL INIT ---
    logger.info("Initializing Fusion Model...")
    model = FusionCodeClassifier(config) 
    model.to(device)

    # Loading Data
    train_dataset, val_dataset, train_sampler, _ = load_data(config, model.tokenizer)

    num_workers = 4
    train_dl = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    val_dl = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    # Optimizer & Scheduler
    grouped_params = get_llrd_optimizer_params(
        model, 
        base_lr=float(config["training"]["learning_rate"]),
        weight_decay=0.01
    )
    optimizer = torch.optim.AdamW(grouped_params)
    scaler = GradScaler()
    
    # SWA Setup
    num_epochs = config["training"].get("num_epochs", 15)
    swa_start_epoch = int(num_epochs * 0.75)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=2e-6)
    
    # [FIX] Inizializziamo lo scheduler PRIMA del loop e UNA VOLTA SOLA
    total_steps = len(train_dl) * num_epochs
    main_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=float(config["training"]["learning_rate"]),
        total_steps=total_steps, 
        pct_start=0.1
    )

    best_f1 = 0.0
    patience = config["training"].get("early_stop_patience", 4)
    patience_counter = 0
    save_dir = config["training"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # --- Training Loop ---
    for epoch in range(num_epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{num_epochs} {'='*20}")
        
        use_swa = (epoch >= swa_start_epoch)
        if use_swa: logger.info(">> SWA Active Phase <<")
        
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, 
            scheduler=None if use_swa else main_scheduler,
            scaler=scaler, device=device, 
            epoch_idx=epoch, total_epochs=num_epochs, 
            swa_model=swa_model if use_swa else None,
            swa_scheduler=swa_scheduler if use_swa else None
        )
        ConsoleUX.log_metrics("Train", train_metrics)
        
        # Validation
        eval_model = swa_model if use_swa else model
        if use_swa: torch.optim.swa_utils.update_bn(train_dl, swa_model, device=device)
        
        val_metrics, val_preds, val_refs = evaluate(eval_model, val_dl, device)
        ConsoleUX.log_metrics("Valid", val_metrics)
        
        if experiment:
            experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
            experiment.log_metrics(val_metrics, prefix="Val", step=epoch)
        
        # Checkpointing
        if val_metrics["f1"] > best_f1:
            logger.info(f"New Best F1: {val_metrics['f1']:.4f}")
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            
            save_path = os.path.join(save_dir, "best_model.pt")
            model_to_save = swa_model.module if use_swa else model
            torch.save(model_to_save.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience and not use_swa:
                logger.info("Early stopping.")
                break

    logger.info("Training Finished.")