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
from dotenv import load_dotenv
from comet_ml import Experiment

from src.src_TaskA.models.model import HybridCodeClassifier
from src.src_TaskA.dataset.dataset import load_vectorized_data 
from src.src_TaskA.utils.utils import evaluate_model, set_seed

# -----------------------------------------------------------------------------
# 1. SETUP & UTILS
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TrainEngine")

class ConsoleUX:
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        log_str = f"[{stage}] "
        keys = ["loss", "accuracy", "f1_macro"]
        for k in keys:
            if k in metrics:
                log_str += f"{k}: {metrics[k]:.4f} | "
        logger.info(log_str.strip(" | "))

def save_checkpoint(model, path, epoch, metrics, config):
    """Saves lightweight model state (no tokenizer needed)."""
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving checkpoint to {path}...")
    torch.save(model.state_dict(), os.path.join(path, "model_state.bin"))
    with open(os.path.join(path, "training_meta.yaml"), "w") as f:
        yaml.dump({"epoch": epoch, "metrics": metrics, "config": config}, f)

# -----------------------------------------------------------------------------
# 2. TRAINING ENGINE
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch_idx):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch_idx+1}", leave=False, dynamic_ncols=True)
    
    for batch in pbar:
        sem_emb = batch["semantic_embedding"].to(device, non_blocking=True)
        struct_feats = batch["structural_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(
                semantic_embedding=sem_emb,
                structural_features=struct_feats,
                labels=labels
            )
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler:
            scheduler.step()
        
        running_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return {"loss": running_loss / len(dataloader)}

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="SemEval Task A - Fast Vectorized Training")
    parser.add_argument("--config", type=str, default="src/src_TaskA/config/config.yaml")    
    parser.add_argument("--holdout_language", type=str, default=None, 
                        help="LOLO Strategy: Language to exclude from training (e.g., 'Python')")
    
    args = parser.parse_args()
    
    ConsoleUX.print_banner("SemEval 2026 - Task A (Fast Vectorized Mode)")
    
    # 1. Config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found at {args.config}")
        sys.exit(1)
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)["common"]
        
    set_seed(config["seed"])
    
    # 2. Comet ML
    api_key = os.getenv("COMET_API_KEY")
    if api_key:
        experiment = Experiment(
            api_key=api_key, 
            project_name=config.get("project_name", "semeval-task-a-ood"),
            auto_metric_logging=False
        )
        experiment.set_name(f"FAST-LOLO-{args.holdout_language}" if args.holdout_language else "FAST-Std-Run")
        experiment.log_parameters(config)
    else:
        experiment = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")

    # 3. Data Loading
    train_dataset, val_dataset = load_vectorized_data(
        config, 
        holdout_language=args.holdout_language
    )
    
    train_dl = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=0,
        drop_last=True
    )
    
    val_dl = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"] * 2, 
        shuffle=False, 
        num_workers=0
    )

    # 4. Model Init
    # Ensure config matches extraction dimensions
    config["semantic_embedding_dim"] = 768 # Standard UniXcoder
    config["structural_feature_dim"] = 10  # Standard Style Engine
    
    logger.info("Initializing Hybrid MLP...")
    model = HybridCodeClassifier(config)
    model.to(device)
    
    # 5. Optimization
    optimizer = AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=1e-2)
    scaler = GradScaler()
    
    total_steps = len(train_dl) * config["num_epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=float(config["learning_rate"]), 
        total_steps=total_steps,
        pct_start=0.1
    )

    # 6. Training Loop
    best_f1 = float("-inf")
    patience = config.get("early_stop_patience", 5)
    patience_counter = 0
    checkpoint_dir = config["checkpoint_dir"]
    
    if args.holdout_language:
        checkpoint_dir = os.path.join(checkpoint_dir, f"holdout_{args.holdout_language}")

    logger.info("Starting Fast Training...")
    
    for epoch in range(config["num_epochs"]):
        # --- TRAIN ---
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, epoch
        )
        
        # --- VALIDATION ---
        val_metrics, val_report = evaluate_model(model, val_dl, device)
        
        # Logging
        ConsoleUX.log_metrics(f"Ep {epoch+1}", {**train_metrics, **val_metrics})
        
        if experiment:
            experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
            experiment.log_metrics(val_metrics, prefix="Val", step=epoch)
            if epoch % 5 == 0: logger.info(f"\n{val_report}")

        # Checkpointing
        current_f1 = val_metrics["f1_macro"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            save_checkpoint(model, os.path.join(checkpoint_dir, "best_model"), epoch, val_metrics, config)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info("Early Stopping Triggered.")
            break
    
    if experiment:
        experiment.end()
        
    logger.info(f"Done. Best F1: {best_f1:.4f}")