import os
import sys
import logging
import yaml
import torch
import numpy as np
import zipfile
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

from comet_ml import Experiment

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Warning: 'kaggle' library not found. Auto-download disabled.")

from src.src_TaskA.models.model import CodeClassifier
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
# Kaggle Data Helper
# -----------------------------------------------------------------------------
def setup_kaggle_data(download_dir="./data"):
    COMPETITION_SLUG = "semeval-2026-task-13-subtask-a" 
    task_dir = os.path.join(download_dir, "Task_A")
    train_file = os.path.join(task_dir, "train.parquet")
    
    if os.path.exists(train_file):
        logger.info(f"Data found in {task_dir}. Skipping download.")
        return task_dir

    if not KAGGLE_AVAILABLE:
        logger.error("Kaggle lib not installed & data missing. Please download manually.")
        sys.exit(1)

    logger.info(f"Downloading data from Kaggle: {COMPETITION_SLUG}...")
    try:
        api = KaggleApi()
        api.authenticate()
        os.makedirs(task_dir, exist_ok=True)
        api.competition_download_files(COMPETITION_SLUG, path=task_dir, quiet=False)
        for item in os.listdir(task_dir):
            if item.endswith(".zip"):
                zip_path = os.path.join(task_dir, item)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(task_dir)
                os.remove(zip_path)
        return task_dir
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        raise e

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
    progress_bar = tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True)
    
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
            "Alpha": f"{alpha:.2f}",
            "LR": f"{scheduler.get_last_lr()[0]:.1e}"
        })

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        predictions.extend(preds)
        references.extend(labels_cpu)
        
        del input_ids, attention_mask, labels, lang_ids, logits, loss

    metrics = model.compute_metrics(predictions, references)
    metrics["loss"] = running_loss / len(dataloader)
    return metrics

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    ConsoleUX.print_banner("SemEval 2026 Task 13 - Subtask A (T4 Optimized)")

    DATA_ROOT = os.getenv("DATA_PATH", "./data")
    task_data_dir = setup_kaggle_data(DATA_ROOT) 
    
    config_path = "src/src_TaskA/config/config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_files = [f for f in os.listdir(task_data_dir) if "train" in f and f.endswith(".parquet")]
    val_files = [f for f in os.listdir(task_data_dir) if ("val" in f or "dev" in f) and f.endswith(".parquet")]

    if train_files:
        config["data"]["train_path"] = os.path.join(task_data_dir, train_files[0])
    if val_files:
        config["data"]["val_path"] = os.path.join(task_data_dir, val_files[0])

    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME"),
        workspace=os.getenv("COMET_WORKSPACE"),
        auto_metric_logging=False
    )
    experiment.log_parameters(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    logger.info("Initializing Model...")
    model = CodeClassifier(config)
    model.to(device)

    train_dataset, val_dataset, _, _ = load_data(config, model.tokenizer)

    num_workers = 4 
    train_dl = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
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

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config["training"]["learning_rate"]),
        weight_decay=0.01
    )
    
    scaler = GradScaler()

    acc_steps = config["training"].get("gradient_accumulation_steps", 1)
    num_epochs = config["training"].get("num_epochs", 10)
    total_steps = len(train_dl) * num_epochs // acc_steps
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=float(config["training"]["learning_rate"]),
        total_steps=total_steps, 
        pct_start=0.1
    )

    best_f1 = 0.0
    patience = config["training"].get("early_stop_patience", 3)
    patience_counter = 0
    save_dir = config["training"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, 
            epoch, num_epochs, acc_steps
        )
        ConsoleUX.log_metrics("Train", train_metrics)
        
        val_metrics, val_preds, val_refs = evaluate(model, val_dl, device)
        ConsoleUX.log_metrics("Valid", val_metrics)
        
        experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
        experiment.log_metrics(val_metrics, prefix="Val", step=epoch)
        
        if val_metrics["f1"] > best_f1:
            logger.info(f"⭐ New Best F1: {val_metrics['f1']:.4f}")
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            
            # --- SALVATAGGIO ---
            save_path = os.path.join(save_dir, "best_model_taskA.pt")
            
            if hasattr(model, "use_lora") and model.use_lora:
                model.base_model.save_pretrained(save_dir)
                
                torch.save({
                    'classifier': model.classifier.state_dict(),
                    'projection': model.projection_head.state_dict(),
                    'language': model.language_classifier.state_dict(),
                    'pooler': model.pooler.state_dict()
                }, os.path.join(save_dir, "heads.pt"))
            else:
                torch.save(model.state_dict(), save_path)
                
            cm = confusion_matrix(val_refs, val_preds)
            experiment.log_confusion_matrix(matrix=cm, title=f"CM_Epoch_{epoch}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    experiment.end()
    logger.info("Training Finished.")