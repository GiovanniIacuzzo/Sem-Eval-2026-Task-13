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
from transformers import AutoTokenizer

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None

from src.src_TaskB.models.model import CodeClassifier
from src.src_TaskB.dataset.dataset import load_data
from src.src_TaskB.utils.utils import evaluate, set_seed

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

LABELS_BINARY = ["Human", "AI"]
LABELS_FAMILIES = [
    "01-ai", "BigCode", "DeepSeek", "Gemma", "Phi", 
    "Llama", "Granite", "Mistral", "Qwen", "OpenAI"
]

class ConsoleUX:
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        log_str = f"[{stage}] "
        keys = ["f1_macro", "f1_weighted", "accuracy", "loss"] + \
               [k for k in metrics.keys() if k not in ["f1_macro", "f1_weighted", "accuracy", "loss"] and "cls" not in k]
        
        for k in keys:
            if k in metrics:
                v = metrics[k]
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                log_str += f"{k}: {val_str} | "
        logger.info(log_str.strip(" | "))

def save_checkpoint(model, path, is_peft=False):
    """
    Salva il modello gestendo correttamente le parti PEFT e Custom.
    """
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving model to {path}...")
    
    model.tokenizer.save_pretrained(path)

    if is_peft:
        model.base_model.save_pretrained(path)
        
        custom_state = {
            'classifier': model.classifier.state_dict(),
            'pooler': model.pooler.state_dict(),
            'projection_head': model.projection_head.state_dict(),
            'language_classifier': model.language_classifier.state_dict()
        }
        torch.save(custom_state, os.path.join(path, "custom_components.pt"))
    else:
        torch.save(model.state_dict(), os.path.join(path, "full_model.bin"))

# -----------------------------------------------------------------------------
# Training
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
        labels_cpu = labels.detach().cpu().numpy()
        predictions.extend(preds)
        references.extend(labels_cpu)

    if hasattr(model, 'compute_metrics'):
        metrics = model.compute_metrics(predictions, references)
    else:
        metrics = {"accuracy": 0.0}
    
    metrics["loss"] = running_loss / len_dataloader
    
    return metrics

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/src_TaskB/config/config.yaml")
    parser.add_argument("--mode", type=str, required=True, choices=["binary", "families"], 
                        help="Choose training mode: 'binary' (Human vs AI) or 'families'")
    args = parser.parse_args()
    
    ConsoleUX.print_banner(f"SemEval 2026 Task 13 - Mode: {args.mode.upper()}")
    
    set_seed(42)

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)

    mode_config = raw_config["common"].copy()
    if args.mode in raw_config:
        mode_config.update(raw_config[args.mode])
    else:
        logger.error(f"Config section for '{args.mode}' not found in yaml!")
        sys.exit(1)
        
    final_config = {
        "model": {
            "model_name": mode_config["model_name"],
            "num_labels": mode_config["num_labels"],
            "use_lora": mode_config.get("use_lora", False),
            "lora_r": mode_config.get("lora_r", 32),
            "lora_dropout": mode_config.get("lora_dropout", 0.1),
            "languages": mode_config.get("languages", []),
            "class_weights": mode_config.get("class_weights", False)
        },
        "training": mode_config,
        "data": mode_config 
    }
    
    current_label_names = LABELS_BINARY if args.mode == "binary" else LABELS_FAMILIES

    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME"),
        workspace=os.getenv("COMET_WORKSPACE"),
        auto_metric_logging=False
    )
    experiment.add_tag(args.mode)
    experiment.log_parameters(mode_config)

    if not torch.cuda.is_available():
        logger.warning("CUDA not found! Training will be slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        logger.info(f"GPU Active: {torch.cuda.get_device_name(0)}")
    
    model_name = mode_config["model_name"]
    logger.info(f"Loading Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Loading Data for mode: {args.mode}...")
    train_dataset, val_dataset, class_weights = load_data(final_config, tokenizer, mode=args.mode)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info(f"Class Weights applied (Size: {class_weights.shape})")
    else:
        logger.info("No Class Weights used (Balanced/Standard Loss).")

    num_workers = 4
    train_dl = DataLoader(
        train_dataset, 
        batch_size=mode_config["batch_size"], 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    val_dl = DataLoader(
        val_dataset, 
        batch_size=mode_config["batch_size"], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    logger.info(f"Initializing Model ({args.mode})...")
    model_wrapper = CodeClassifier(final_config, class_weights=class_weights)
    model_wrapper.to(device)

    is_peft = final_config["model"]["use_lora"]
    if is_peft:
        trainable_params = sum(p.numel() for p in model_wrapper.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model_wrapper.parameters())
        logger.info(f"LoRA Active. Trainable: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_wrapper.parameters()), 
        lr=float(mode_config["learning_rate"]),
        weight_decay=0.01
    )
    
    scaler = GradScaler()
    
    acc_steps = mode_config["gradient_accumulation_steps"]
    num_epochs = mode_config["num_epochs"]
    total_steps = len(train_dl) * num_epochs // acc_steps
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(mode_config["learning_rate"]),
        total_steps=total_steps,
        pct_start=0.1
    )

    best_f1 = float("-inf")
    patience = mode_config.get("early_stop_patience", 4)
    patience_counter = 0

    checkpoint_dir = os.path.join(mode_config["checkpoint_dir"], args.mode)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("Starting Training...")
    
    for epoch in range(num_epochs):
        ConsoleUX.print_banner(f"Epoch {epoch+1}/{num_epochs}")

        train_metrics = train_one_epoch(
            model_wrapper, train_dl, optimizer, scheduler, scaler, device, 
            epoch, num_epochs, acc_steps
        )
        ConsoleUX.log_metrics("Train", train_metrics)
        experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
        torch.cuda.empty_cache() 
        
        try:
            val_metrics, val_preds, val_refs, val_report = evaluate(
                model_wrapper, val_dl, device, label_names=current_label_names
            )
        except ValueError:
            val_metrics, val_preds, val_refs = evaluate(model_wrapper, val_dl, device)
            val_report = "Report not available"
        
        ConsoleUX.log_metrics("Valid", val_metrics)
        experiment.log_metrics(val_metrics, prefix="Val", step=epoch)

        current_f1 = val_metrics.get("f1_macro", 0.0)
        
        if current_f1 > best_f1:
            logger.info(f"New Best Macro F1: {current_f1:.4f} (prev: {best_f1:.4f})")
            best_f1 = current_f1
            patience_counter = 0
            
            save_checkpoint(model_wrapper, checkpoint_dir, is_peft=is_peft)
            
            try:
                cm = confusion_matrix(val_refs, val_preds)
                if len(cm) == len(current_label_names):
                    experiment.log_confusion_matrix(
                        matrix=cm, 
                        title=f"CM_{args.mode}_Best",
                        labels=current_label_names,
                        file_name=f"confusion_matrix_epoch_{epoch}.json"
                    )
            except Exception as e:
                logger.warning(f"CM Log Error: {e}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    experiment.end()
    logger.info(f"Training Complete for mode: {args.mode}")