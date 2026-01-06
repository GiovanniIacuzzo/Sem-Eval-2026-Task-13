import os
import sys
import yaml
import json
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

from src.src_TaskB.models.model import CodeClassifier
from src.src_TaskB.dataset.dataset import load_data
from src.src_TaskB.utils.utils import set_seed, evaluate_model

# -----------------------------------------------------------------------------
# 1. SETUP & UTILS
# -----------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_label_mapping(mode, data_dir):
    """Carica il mapping delle label per il logging e la confusion matrix."""
    if mode == "binary":
        return ["Human", "AI"]
    
    mapping_path = os.path.join(data_dir, "family_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        # Ordina le chiavi in base ai valori (0, 1, 2...)
        sorted_labels = [k for k, v in sorted(mapping.items(), key=lambda item: int(item[1]))]
        logger.info(f"Loaded Dynamic Labels: {sorted_labels}")
        return sorted_labels
    else:
        logger.warning("Mapping file not found! Falling back to generic labels.")
        return [f"Class_{i}" for i in range(11)]

class ConsoleUX:
    """Utility per output console puliti e professionali."""
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        log_str = f"[{stage}] "
        keys = ["loss", "f1_macro", "accuracy", "task_loss", "con_loss"]
        for k in keys:
            if k in metrics:
                log_str += f"{k}: {metrics[k]:.4f} | "
        logger.info(log_str.strip(" | "))

def save_checkpoint(model, tokenizer, path, epoch, metrics, is_peft=False):
    """Salva modello, tokenizer e metadati."""
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving checkpoint to {path}...")
    
    tokenizer.save_pretrained(path)
    
    if is_peft:
        # Se usi LoRA, salva solo gli adattatori e la classifier head
        model.base_model.save_pretrained(path)
        torch.save(model.classifier.state_dict(), os.path.join(path, "classifier.pt"))
    else:
        # Full Fine-Tuning
        torch.save(model.state_dict(), os.path.join(path, "model_state.bin"))
    
    # Salviamo i metadati
    with open(os.path.join(path, "training_meta.yaml"), "w") as f:
        yaml.dump({"epoch": epoch, "metrics": metrics}, f)

# -----------------------------------------------------------------------------
# 2. TRAINING ENGINE
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, 
                   epoch_idx, acc_steps=1, contrastive_fn=None):
    model.train()
    
    running_metrics = {"loss": 0.0, "task_loss": 0.0, "con_loss": 0.0}
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch_idx+1}", leave=False, dynamic_ncols=True)
    
    optimizer.zero_grad(set_to_none=True)
    
    for step, batch in enumerate(pbar):
        # Spostamento dati su Device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        extra_features = batch.get("extra_features", None)
        if extra_features is not None:
            extra_features = extra_features.to(device, non_blocking=True)
        
        # Mixed Precision Forward
        with autocast(device_type='cuda', dtype=torch.float16):
            # Il modello restituisce: logits, task_loss (Focal), proj_features
            _, loss_task, proj_features = model(
                input_ids, attention_mask, labels=labels, extra_features=extra_features
            )
            
            # Calcolo Loss Contrastiva (se abilitata e siamo in mode 'families')
            loss_con = torch.tensor(0.0, device=device)
            if contrastive_fn is not None:
                # proj_features sono già normalizzate (L2) dal modello
                loss_con = contrastive_fn(proj_features, labels)
            
            # Combinazione Loss (Task + 0.5 * Contrastive)
            total_loss = (loss_task + 0.5 * loss_con) / acc_steps

        # Backward
        scaler.scale(total_loss).backward()

        # Gradient Accumulation Step
        if (step + 1) % acc_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if scheduler is not None:
                scheduler.step()
        
        # Logging
        current_loss = total_loss.item() * acc_steps
        running_metrics["loss"] += current_loss
        running_metrics["task_loss"] += loss_task.item()
        running_metrics["con_loss"] += loss_con.item()
        
        pbar.set_postfix({
            "L": f"{current_loss:.3f}",
            "T": f"{loss_task.item():.3f}",
            "C": f"{loss_con.item():.3f}" if contrastive_fn else "0.0"
        })

    # Calcolo medie
    steps = len(dataloader)
    return {k: v / steps for k, v in running_metrics.items()}

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="SemEval Task B Training")
    parser.add_argument("--config", type=str, default="src/src_TaskB/config/config.yaml")
    parser.add_argument("--mode", type=str, required=True, choices=["binary", "families"], 
                        help="Choose training mode: binary detection or family classification")
    args = parser.parse_args()
    
    ConsoleUX.print_banner(f"SemEval Task 13 - Subtask B [{args.mode.upper()}]")
    
    # 1. Load Configuration
    if not os.path.exists(args.config):
        logger.error(f"Config file not found at {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)

    config = raw_config["common"].copy()
    if args.mode in raw_config:
        config.update(raw_config[args.mode])
    
    set_seed(config.get("seed", 42))

    # 2. Setup Comet ML
    api_key = os.getenv("COMET_API_KEY")
    if api_key:
        experiment = Experiment(
            api_key=api_key,
            project_name=os.getenv("COMET_PROJECT_NAME", "semeval-task-b"),
            auto_metric_logging=False
        )
        experiment.add_tag(args.mode)
        experiment.log_parameters(config)
    else:
        logger.warning("COMET_API_KEY not found. Logging will be local only.")
        experiment = None

    # 3. Dynamic Setup
    labels_list = load_label_mapping(args.mode, config.get("data_dir", "data/Task_B_Processed"))
    
    model_config = {
        "model": {
            "model_name": config["model_name"],
            "num_labels": len(labels_list),
            "num_extra_features": config.get("num_extra_features", 8),
            "use_lora": config.get("use_lora", False),
        },
        "training": config,
        "data": config 
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device} | Labels: {len(labels_list)}")

    # 4. Tokenizer & Data
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    train_dataset, val_dataset, class_weights = load_data(model_config, tokenizer, mode=args.mode)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info(f"Class Weights loaded: {class_weights}")

    train_dl = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config.get("num_workers", 4),
        pin_memory=True, 
        persistent_workers=True,
        drop_last=True
    )
    val_dl = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"]*2, 
        shuffle=False, 
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )

    # 5. Model Init
    model = CodeClassifier(model_config, class_weights=class_weights)
    model.to(device)

    # 6. Optimizer & Scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=float(config["learning_rate"]), 
        weight_decay=config.get("weight_decay", 0.01)
    )
    
    scaler = GradScaler()
    
    acc_steps = config.get("gradient_accumulation_steps", 1)
    total_steps = (len(train_dl) // acc_steps) * config["num_epochs"]
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=float(config["learning_rate"]), 
        total_steps=total_steps,
        pct_start=0.1
    )

    # 7. Setup Contrastive Loss
    contrastive_fn = None
    if args.mode == "families":
        logger.info("Initializing SupConLoss (Pytorch Metric Learning)...")
        contrastive_fn = losses.SupConLoss(temperature=0.1)
        contrastive_fn = contrastive_fn.to(device)

    # 8. Training Loop
    best_f1 = float("-inf")
    patience_counter = 0
    patience = config.get("early_stop_patience", 3)
    checkpoint_dir = os.path.join(config["checkpoint_dir"], args.mode)

    logger.info("Starting Training...")

    for epoch in range(config["num_epochs"]):
        ConsoleUX.print_banner(f"Epoch {epoch+1}/{config['num_epochs']}")

        # --- TRAIN ---
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, 
            epoch, acc_steps, contrastive_fn
        )
        ConsoleUX.log_metrics("Train", train_metrics)
        if experiment:
            experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
            experiment.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

        # --- VALIDATION ---
        val_metrics, val_preds, val_refs, val_report = evaluate_model(model, val_dl, device, label_names=labels_list)
        
        ConsoleUX.log_metrics("Valid", val_metrics)
        logger.info(f"\n{val_report}")
        if experiment:
            experiment.log_metrics(val_metrics, prefix="Val", step=epoch)

        # --- CHECKPOINT & EARLY STOPPING ---
        current_f1 = val_metrics.get("f1_macro", 0.0)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            
            save_path = os.path.join(checkpoint_dir, "best_model")
            save_checkpoint(
                model, tokenizer, save_path, epoch, val_metrics, 
                is_peft=model_config["model"]["use_lora"]
            )
            
            cm = confusion_matrix(val_refs, val_preds)
            if experiment:
                experiment.log_confusion_matrix(matrix=cm, title=f"CM_{args.mode}", labels=labels_list)
                experiment.log_metric("best_f1", best_f1, step=epoch)
                
            logger.info(f"---> New Best F1: {best_f1:.4f}. Model Saved.")
        else:
            patience_counter += 1
            logger.warning(f"---> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            ConsoleUX.print_banner("EARLY STOPPING TRIGGERED")
            break
        
    if experiment:
        experiment.end()
        
    logger.info("Training Completed Successfully.")