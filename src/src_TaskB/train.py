import os
import sys
import json
import logging
import yaml
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler 
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from comet_ml import Experiment
from transformers import AutoTokenizer

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None

from src.src_TaskB.models.model import CodeClassifier
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
    """Carica le etichette reali dal file JSON generato in preprocessing"""
    if mode == "binary":
        return ["Human", "AI"]
    
    mapping_path = os.path.join(data_dir, "family_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)

        sorted_labels = [v['generator'] for k, v in sorted(mapping.items(), key=lambda item: int(item[0]))]
        logger.info(f"Loaded Dynamic Labels: {sorted_labels}")
        return sorted_labels
    else:
        logger.warning("Mapping file not found! Falling back to generic labels.")
        return [f"Class_{i}" for i in range(10)]

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
                   epoch_idx, total_epochs, accumulation_steps=1):
    
    model.train()
    running_loss = 0.0
    predictions, references = [], []
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}", leave=False, dynamic_ncols=True)
    len_dataloader = len(dataloader)
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        extra_features = batch.get("extra_features", None)
        if extra_features is not None:
            extra_features = extra_features.to(device, non_blocking=True)
        
        lang_ids = batch.get("lang_ids", None)
        if lang_ids is not None:
            lang_ids = lang_ids.to(device, non_blocking=True)
        
        current_step = step + epoch_idx * len_dataloader
        total_steps = total_epochs * len_dataloader
        p = float(current_step) / (total_steps + 1e-8)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        with autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(
                input_ids, 
                attention_mask, 
                lang_ids=lang_ids, 
                labels=labels, 
                extra_features=extra_features,
                alpha=alpha
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
        predictions.extend(preds)
        references.extend(labels.detach().cpu().numpy())

    # Metriche veloci in-train
    metrics = {
        "accuracy": accuracy_score(references, predictions),
        "f1_macro": f1_score(references, predictions, average="macro"),
        "loss": running_loss / len_dataloader
    }
    
    return metrics

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

    # 1. Load Config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)

    mode_config = raw_config["common"].copy()
    if args.mode in raw_config:
        mode_config.update(raw_config[args.mode])
    
    # 2. Setup Labels Dinamici
    labels_list = load_label_mapping(args.mode, mode_config.get("data_dir", "data/Task_B_Processed"))
    mode_config["num_labels"] = len(labels_list)
    logger.info(f"Detected {mode_config['num_labels']} labels.")

    # 3. Final Config Structure
    final_config = {
        "model": {
            "model_name": mode_config["model_name"],
            "num_labels": mode_config["num_labels"],
            "num_extra_features": 5,
            "use_lora": mode_config.get("use_lora", False),
            "lora_r": mode_config.get("lora_r", 32),
            "lora_dropout": mode_config.get("lora_dropout", 0.1),
            "class_weights": mode_config.get("class_weights", False)
        },
        "training": mode_config,
        "data": mode_config 
    }

    # 4. Comet ML
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME"),
        workspace=os.getenv("COMET_WORKSPACE"),
        auto_metric_logging=False
    )
    experiment.add_tag(args.mode)
    experiment.log_parameters(mode_config)

    # 5. Device & Tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(mode_config["model_name"])

    # 6. Data Loading
    logger.info(f"Loading Data for mode: {args.mode}...")
    train_dataset, val_dataset, class_weights = load_data(final_config, tokenizer, mode=args.mode)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info(f"Class Weights Loaded: {class_weights}")

    train_dl = DataLoader(
        train_dataset, 
        batch_size=mode_config["batch_size"], 
        shuffle=True,
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    val_dl = DataLoader(
        val_dataset, 
        batch_size=mode_config["batch_size"], 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # 7. Model Init
    logger.info("Initializing Enhanced Model...")
    model_wrapper = CodeClassifier(final_config, class_weights=class_weights)
    model_wrapper.to(device)

    # 8. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_wrapper.parameters()), 
        lr=float(mode_config["learning_rate"]),
        weight_decay=0.01
    )
    
    scaler = GradScaler()
    acc_steps = mode_config.get("gradient_accumulation_steps", 1)
    num_epochs = mode_config["num_epochs"]
    total_steps = len(train_dl) * num_epochs // acc_steps
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(mode_config["learning_rate"]),
        total_steps=total_steps,
        pct_start=0.1
    )

    # 9. Training Loop
    best_f1 = float("-inf")
    patience = mode_config.get("early_stop_patience", 4)
    patience_counter = 0
    checkpoint_dir = os.path.join(mode_config["checkpoint_dir"], args.mode)

    logger.info("Starting Training...")
    
    for epoch in range(num_epochs):
        ConsoleUX.print_banner(f"Epoch {epoch+1}/{num_epochs}")

        # Train
        train_metrics = train_one_epoch(
            model_wrapper, train_dl, optimizer, scheduler, scaler, device, 
            epoch, num_epochs, acc_steps
        )
        ConsoleUX.log_metrics("Train", train_metrics)
        experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
        torch.cuda.empty_cache() 
        
        # Val
        val_metrics, val_preds, val_refs, val_report = evaluate_model(
            model_wrapper, val_dl, device, label_names=labels_list
        )
        
        ConsoleUX.log_metrics("Valid", val_metrics)
        experiment.log_metrics(val_metrics, prefix="Val", step=epoch)
        logger.info(f"\n{val_report}")

        # Checkpointing
        current_f1 = val_metrics.get("f1_macro", 0.0)
        if current_f1 > best_f1:
            logger.info(f"New Best F1: {current_f1:.4f}")
            best_f1 = current_f1
            patience_counter = 0
            save_checkpoint(model_wrapper, checkpoint_dir, is_peft=final_config["model"]["use_lora"])
            
            cm = confusion_matrix(val_refs, val_preds)
            experiment.log_confusion_matrix(
                matrix=cm, 
                title=f"CM_{args.mode}_Epoch{epoch}",
                labels=labels_list,
                file_name=f"confusion_matrix_epoch_{epoch}.json"
            )
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    experiment.end()
    logger.info(f"Training Complete. Best Model in {checkpoint_dir}")