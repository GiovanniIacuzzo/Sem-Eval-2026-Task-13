import os
import sys
import logging
import yaml
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer
from comet_ml import Experiment

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from src.src_TaskC.models.model import CodeClassifier
from src.src_TaskC.dataset.dataset import CodeDataset, load_data_for_training, get_dynamic_language_map, get_class_weights
from src.src_TaskC.utils.utils import evaluate

# -----------------------------------------------------------------------------
# Logger & UX Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt=" %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConsoleUX:
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        keys = sorted(metrics.keys(), key=lambda x: (0 if x == 'f1' else 1, x))
        log_str = f"[{stage}] "
        for k in keys:
            v = metrics[k]
            if "class" in k: 
                log_str += f"{k.replace('f1_class_', 'C')}: {v:.3f} | "
            elif isinstance(v, float):
                log_str += f"{k}: {v:.4f} | "
            else:
                log_str += f"{k}: {v} | "
        logger.info(log_str.strip(" | "))

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
    progress_bar = tqdm(dataloader, desc=f"Ep {epoch_idx+1} Train", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        style_feats = batch["style_feats"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        lang_ids = batch["lang_ids"].to(device, non_blocking=True)
        
        current_step = step + epoch_idx * len_dataloader
        total_steps = total_epochs * len_dataloader
        p = float(current_step) / total_steps
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        with autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(
                input_ids, attention_mask, 
                style_feats=style_feats,
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
            "LR": f"{scheduler.get_last_lr()[0]:.1e}"
        })

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        predictions.extend(preds)
        references.extend(labels.detach().cpu().numpy())
    
    metrics = model.compute_metrics(predictions, references)
    metrics["loss"] = running_loss / len_dataloader
    return metrics

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    ConsoleUX.print_banner("SemEval 2026 - Task 13 - Subtask C (Optimized)")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/src_TaskC/config/config.yaml") 
    args = parser.parse_args()

    # --- 1. SETUP CONFIG & DEVICE ---
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # =========================================================================
    # 2. SMART DATA LOADING
    # =========================================================================
    full_df = load_data_for_training(config)
    
    train_df = full_df.sample(frac=0.85, random_state=42)
    val_df = full_df.drop(train_df.index)
    
    language_map = get_dynamic_language_map(full_df)
    config["model"]["num_languages"] = len(language_map)
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    class_weights = get_class_weights(train_df, device)
    
    train_dataset = CodeDataset(train_df, tokenizer, language_map, config["data"]["max_length"], augment=True)
    val_dataset = CodeDataset(val_df, tokenizer, language_map, config["data"]["max_length"], augment=False)

    train_dl = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], 
                         shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], 
                       shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Final Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    # =========================================================================
    # 3. MODEL & EXPERIMENT SETUP
    # =========================================================================
    model = CodeClassifier(config, class_weights=class_weights)
    model.to(device)

    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME", "semeval-task13-subtaskc"),
        workspace=os.getenv("COMET_WORKSPACE"),
        auto_metric_logging=False
    )
    experiment.log_parameters(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=0.01)
    scaler = GradScaler()
    
    acc_steps = config["training"].get("gradient_accumulation_steps", 1)
    num_epochs = config["training"].get("num_epochs", 5) 
    total_steps = len(train_dl) * num_epochs // acc_steps
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(config["training"]["learning_rate"]),
        total_steps=total_steps, pct_start=0.1
    )

    # =========================================================================
    # 4. TRAINING LOOP
    # =========================================================================
    best_f1 = 0.0
    patience = config["training"].get("early_stop_patience", 4)
    patience_counter = 0
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, 
            epoch, num_epochs, acc_steps
        )
        
        val_metrics, _, _ = evaluate(model, val_dl, device, verbose=False)
        
        ConsoleUX.log_metrics(f"Ep{epoch+1}", val_metrics)
        experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
        experiment.log_metrics(val_metrics, prefix="Val", step=epoch)

        if val_metrics["f1"] > best_f1:
            logger.info(f"New Best F1: {val_metrics['f1']:.4f}")
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            
            save_path = os.path.join(checkpoint_dir, "best_model.pt")
            if model.use_lora:
                model.base_model.save_pretrained(checkpoint_dir)
                torch.save(model.classifier.state_dict(), os.path.join(checkpoint_dir, "head.pt"))
            else:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    experiment.end()
    logger.info("Training Finished.")