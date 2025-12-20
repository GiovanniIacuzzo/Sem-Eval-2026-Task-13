import os
import sys
import logging
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

from comet_ml import Experiment

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
# OPTIMIZER UTILS: LLRD (Layer-wise Learning Rate Decay)
# -----------------------------------------------------------------------------
def get_llrd_optimizer_params(model, base_lr, weight_decay=0.01, decay_factor=0.95):
    """
    Imposta learning rate decrescenti: alti per classifier/top layers, bassi per embeddings.
    Cruciale per Full Fine-Tuning su GPU potenti.
    """
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
        logger.warning("Struttura modello non standard per LLRD. Fallback a LR piatto.")
        backbone_params = [p for n, p in named_parameters if "base_model" in n]
        opt_parameters.append({"params": backbone_params, "lr": base_lr * 0.8, "weight_decay": weight_decay})

    return opt_parameters

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
    total_steps_all = len_dataloader * total_epochs 
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        lang_ids = batch["lang_ids"].to(device, non_blocking=True)

        current_global_step = step + epoch_idx * len_dataloader
        p = float(current_global_step) / total_steps_all
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
    ConsoleUX.print_banner("SemEval 2026 Task 13 - Subtask A (Local Data)")

    DATA_ROOT = os.getenv("DATA_PATH", "./data")
    task_data_dir = os.path.join(DATA_ROOT, "Task_A")
    
    if not os.path.exists(task_data_dir):
        logger.error(f"Directory not found: {task_data_dir}")
        logger.error("Please ensure dataset is in ./data/Task_A/")
        sys.exit(1)
    
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
        logger.info(f"Train file: {config['data']['train_path']}")
    else:
        logger.error("No train.parquet found in Task_A folder!")
        sys.exit(1)

    if val_files:
        config["data"]["val_path"] = os.path.join(task_data_dir, val_files[0])
        logger.info(f"Val file: {config['data']['val_path']}")

    api_key = os.getenv("COMET_API_KEY")
    if api_key:
        experiment = Experiment(
            api_key=api_key,
            project_name=os.getenv("COMET_PROJECT_NAME"),
            workspace=os.getenv("COMET_WORKSPACE"),
            auto_metric_logging=False
        )
        experiment.log_parameters(config)
    else:
        logger.warning("Comet API Key not found. Logging disabled.")
        experiment = None

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

    logger.info("Configuring LLRD Optimizer...")
    grouped_params = get_llrd_optimizer_params(
        model, 
        base_lr=float(config["training"]["learning_rate"]),
        weight_decay=0.01,
        decay_factor=0.9
    )
    optimizer = torch.optim.AdamW(grouped_params)
    
    scaler = GradScaler()

    acc_steps = config["training"].get("gradient_accumulation_steps", 1)
    num_epochs = config["training"].get("num_epochs", 10)
    
    total_steps = (len(train_dl) // acc_steps) * num_epochs
    
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

    logger.info(f"Starting Training for {num_epochs} Epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{num_epochs} {'='*20}")
        
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, 
            epoch, num_epochs, acc_steps
        )
        ConsoleUX.log_metrics("Train", train_metrics)
        
        val_metrics, val_preds, val_refs = evaluate(model, val_dl, device)
        ConsoleUX.log_metrics("Valid", val_metrics)
        
        if experiment:
            experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
            experiment.log_metrics(val_metrics, prefix="Val", step=epoch)
        
        if val_metrics["f1"] > best_f1:
            logger.info(f"New Best F1: {val_metrics['f1']:.4f} (Prev: {best_f1:.4f})")
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            
            save_path = os.path.join(save_dir, "best_model_taskA.pt")
            
            if hasattr(model, "use_lora") and model.use_lora:
                model.base_model.save_pretrained(save_dir)
                torch.save({
                    'classifier': model.classifier.state_dict(),
                    'language': model.language_classifier.state_dict(),
                    'pooler': model.pooler.state_dict()
                }, os.path.join(save_dir, "heads.pt"))
                logger.info("Saved LoRA adapters + Custom Heads.")
            else:
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved Full Model to {save_path}")
                
            cm = confusion_matrix(val_refs, val_preds)
            if experiment:
                experiment.log_confusion_matrix(matrix=cm, title=f"CM_Epoch_{epoch}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            logger.info("Early stopping triggered. Exiting.")
            break

    if experiment:
        experiment.end()
    logger.info("Training Finished.")