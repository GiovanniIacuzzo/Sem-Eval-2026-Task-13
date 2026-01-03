import os
import sys
import json
import logging
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler 
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix
from comet_ml import Experiment
from transformers import AutoTokenizer

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None

from src.src_TaskB.models.model import CodeClassifier, SupConLoss
from src.src_TaskB.dataset.dataset import load_data
from src.src_TaskB.utils.utils import set_seed, evaluate_model

# -----------------------------------------------------------------------------
# Configuration & Setup
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

def load_label_mapping(mode, data_dir):
    if mode == "binary":
        return ["Human", "AI"]
    
    mapping_path = os.path.join(data_dir, "family_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        sorted_labels = [k for k, v in sorted(mapping.items(), key=lambda item: int(item[1]))]
        logger.info(f"Loaded Dynamic Labels: {sorted_labels}")
        return sorted_labels
    else:
        logger.warning("Mapping file not found! Falling back to generic labels.")
        return [f"Class_{i}" for i in range(11)]

class ConsoleUX:
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        log_str = f"[{stage}] "
        keys = ["f1_macro", "f1_weighted", "accuracy", "loss"]
        for k in keys:
            if k in metrics:
                log_str += f"{k}: {metrics[k]:.4f} | "
        logger.info(log_str.strip(" | "))

def save_checkpoint(model, path, is_peft=False):
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving model to {path}...")
    model.tokenizer.save_pretrained(path)
    if is_peft:
        model.base_model.save_pretrained(path)
        torch.save(model.classifier.state_dict(), os.path.join(path, "classifier.pt"))
    else:
        torch.save(model.state_dict(), os.path.join(path, "full_model.bin"))

# -----------------------------------------------------------------------------
# Training Engine
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, 
                   epoch_idx, total_epochs, accumulation_steps=1, contrastive_fn=None):
    model.train()
    running_loss = 0.0
    running_task_loss = 0.0
    running_con_loss = 0.0
    
    optimizer.zero_grad()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        extra_features = batch.get("extra_features", None)
        if extra_features is not None:
            extra_features = extra_features.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            logits, loss_task, proj_features = model(
                input_ids, attention_mask, labels=labels, extra_features=extra_features
            )
            
            loss_con = torch.tensor(0.0).to(device)
            if contrastive_fn is not None:
                loss_con = contrastive_fn(proj_features, labels)
            
            total_loss = (loss_task + 0.5 * loss_con) / accumulation_steps

        scaler.scale(total_loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        running_loss += total_loss.item() * accumulation_steps
        running_task_loss += loss_task.item()
        running_con_loss += loss_con.item()
        
        progress_bar.set_postfix({
            "Loss": f"{total_loss.item()*accumulation_steps:.3f}",
            "Task": f"{loss_task.item():.3f}",
            "Con": f"{loss_con.item():.2f}" if contrastive_fn else "N/A"
        })

    return {
        "loss": running_loss / len(dataloader),
        "task_loss": running_task_loss / len(dataloader),
        "con_loss": running_con_loss / len(dataloader)
    }

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/src_TaskB/config/config.yaml")
    parser.add_argument("--mode", type=str, required=True, choices=["binary", "families"])
    args = parser.parse_args()
    
    ConsoleUX.print_banner("SemEval 2026 - Task 13 - subtask B")
    set_seed(42)

    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)

    mode_config = raw_config["common"].copy()
    if args.mode in raw_config:
        mode_config.update(raw_config[args.mode])
    
    labels_list = load_label_mapping(args.mode, mode_config.get("data_dir", "data/Task_B_Processed"))
    mode_config["num_labels"] = len(labels_list)

    final_config = {
        "model": {
            "model_name": mode_config["model_name"],
            "num_labels": mode_config["num_labels"],
            "num_extra_features": 8,
            "use_lora": mode_config.get("use_lora", False),
            "lora_r": mode_config.get("lora_r", 32),
            "lora_dropout": mode_config.get("lora_dropout", 0.1),
            "class_weights": mode_config.get("class_weights", True)
        },
        "training": mode_config,
        "data": mode_config 
    }

    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME"),
        auto_metric_logging=False
    )
    experiment.add_tag(args.mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(mode_config["model_name"])

    # Caricamento dataset
    train_dataset, val_dataset, class_weights = load_data(final_config, tokenizer, mode=args.mode)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # DATALOADER OTTIMIZZATI
    train_dl = DataLoader(
        train_dataset, 
        batch_size=mode_config["batch_size"], 
        shuffle=True, 
        num_workers=8,
        pin_memory=True, 
        persistent_workers=True,
        drop_last=True
    )
    val_dl = DataLoader(val_dataset, batch_size=mode_config["batch_size"]*2, shuffle=False, num_workers=4, pin_memory=True)

    model_wrapper = CodeClassifier(final_config, class_weights=class_weights)
    model_wrapper.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_wrapper.parameters()), lr=float(mode_config["learning_rate"]), weight_decay=0.01)
    
    scaler = GradScaler()
    acc_steps = mode_config.get("gradient_accumulation_steps", 1)
    total_steps = len(train_dl) * mode_config["num_epochs"] // acc_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(mode_config["learning_rate"]), total_steps=total_steps)

    contrastive_fn = SupConLoss(temperature=0.1) if args.mode == "families" else None

    best_f1 = float("-inf")
    patience = mode_config.get("early_stop_patience", 3)
    patience_counter = 0
    checkpoint_dir = os.path.join(mode_config["checkpoint_dir"], args.mode)

    for epoch in range(mode_config["num_epochs"]):
        ConsoleUX.print_banner(f"Epoch {epoch+1}/{mode_config['num_epochs']}")

        # Training
        train_metrics = train_one_epoch(
            model_wrapper, train_dl, optimizer, scheduler, scaler, device, 
            epoch, mode_config["num_epochs"], acc_steps, contrastive_fn=contrastive_fn
        )
        experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
        
        # Valutazione
        val_metrics, val_preds, val_refs, val_report = evaluate_model(model_wrapper, val_dl, device, label_names=labels_list)
        
        ConsoleUX.log_metrics("Valid", val_metrics)
        experiment.log_metrics(val_metrics, prefix="Val", step=epoch)
        logger.info(f"\n{val_report}")

        current_f1 = val_metrics.get("f1_macro", 0.0)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            save_checkpoint(model_wrapper, checkpoint_dir, is_peft=final_config["model"]["use_lora"])
            
            cm = confusion_matrix(val_refs, val_preds)
            experiment.log_confusion_matrix(matrix=cm, title=f"CM_{args.mode}", labels=labels_list)
            logger.info(f"---> New Best F1: {best_f1:.4f}. Model Saved.")
        else:
            patience_counter += 1
            logger.warning(f"---> No improvement. Early Stopping: {patience_counter}/{patience}")

        if patience_counter >= patience:
            ConsoleUX.print_banner("EARLY STOPPING TRIGGERED")
            break
        
    experiment.end()