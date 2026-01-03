import os
import sys
import yaml
import torch
import argparse
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler 
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix
from comet_ml import Experiment
from transformers import AutoTokenizer

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None

from src.src_TaskA.models.model import CodeModel
from src.src_TaskA.dataset.dataset import load_data
from src.src_TaskA.utils.utils import set_seed, evaluate_model, ConsoleUX

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

def save_checkpoint(model, path, tokenizer):
    """
    Salvataggio ottimizzato per LoRA:
    1. Salva i pesi LoRA (base_model)
    2. Salva la head di classificazione custom (classifier)
    3. Salva il tokenizer
    """
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving model to {path}...")
    
    tokenizer.save_pretrained(path)
    
    model.base_model.save_pretrained(path)
    
    torch.save(model.classifier.state_dict(), os.path.join(path, "classifier_head.pt"))
    torch.save(model.extra_projector.state_dict(), os.path.join(path, "projector.pt"))

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, 
                   epoch_idx, total_epochs, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    
    optimizer.zero_grad()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}/{total_epochs}", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        extra_features = batch.get("extra_features", None).to(device, non_blocking=True)
        
        dtype_amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        with autocast(device_type='cuda', dtype=dtype_amp):
            logits, loss, _ = model(
                input_ids, attention_mask, labels=labels, extra_features=extra_features
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
        
        running_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix({"Loss": f"{loss.item()*accumulation_steps:.4f}"})

    return {"loss": running_loss / len(dataloader)}

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/src_TaskA/config/config.yaml")
    args = parser.parse_args()
    
    ConsoleUX.print_banner("SemEval 2026 - Task 13 - Subtask A")
    set_seed(42)

    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)["common"]

    # Comet ML Setup
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME"),
        auto_metric_logging=False
    )
    experiment.set_name(f"TaskA_StarCoder_{config['model_name'].split('/')[-1]}")
    experiment.log_parameters(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")

    # Tokenizer & Data
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    train_dataset, val_dataset, class_weights = load_data(config, tokenizer)
    
    # Task A: Human (0) vs AI (1)
    label_names = ["Human", "AI"]
    config["num_labels"] = 2
    
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # Dataloaders
    train_dl = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    val_dl = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # Model Setup
    model = CodeModel(config, class_weights=class_weights)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config["learning_rate"]))
    scaler = GradScaler()
    
    acc_steps = config.get("gradient_accumulation_steps", 1)
    total_steps = len(train_dl) * config["num_epochs"] // acc_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(config["learning_rate"]), total_steps=total_steps)

    # Training Loop
    best_f1 = 0.0
    patience_counter = 0
    patience = config.get("early_stop_patience", 3)
    checkpoint_dir = config["checkpoint_dir"]

    for epoch in range(config["num_epochs"]):
        ConsoleUX.print_banner(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, 
            epoch, config["num_epochs"], acc_steps
        )
        experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Validation
        val_metrics, val_preds, val_refs, val_report = evaluate_model(model, val_dl, device, label_names=label_names)
        
        ConsoleUX.log_metrics("Valid", val_metrics)
        experiment.log_metrics(val_metrics, prefix="Val", step=epoch)
        logger.info(f"\n{val_report}")

        current_f1 = val_metrics["f1_macro"]
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            
            # Save Checkpoint
            save_path = os.path.join(checkpoint_dir, "best_model")
            save_checkpoint(model, save_path, tokenizer)
            
            cm = confusion_matrix(val_refs, val_preds)
            experiment.log_confusion_matrix(
                matrix=cm, 
                title=f"Confusion Matrix Epoch {epoch}", 
                labels=label_names
            )
            logger.info(f"--> New Best F1: {best_f1:.4f}. Model Saved.")
        else:
            patience_counter += 1
            logger.warning(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                logger.warning("Early Stopping Triggered")
                break
    
    experiment.end()

if __name__ == "__main__":
    main()