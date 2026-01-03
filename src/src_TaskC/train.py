import os
import sys
import logging
import yaml
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler # Updated import
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer
from comet_ml import Experiment

# Optimization settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from src.src_TaskC.models.model import CodeClassifier
from src.src_TaskC.dataset.dataset import CodeDataset, load_data_for_training, get_class_weights
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
                # Abbreviate class metrics for readability
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
    correct_preds = 0
    total_preds = 0
    
    optimizer.zero_grad()
    len_dataloader = len(dataloader)
    
    # Calculate total optimization steps for alpha scheduling
    total_opt_steps = (len_dataloader // accumulation_steps) * total_epochs
    current_opt_step = (epoch_idx * (len_dataloader // accumulation_steps))
    
    progress_bar = tqdm(dataloader, desc=f"Ep {epoch_idx+1} Train", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(progress_bar):
        # Move to device (non_blocking=True for speed)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        style_feats = batch["style_feats"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        lang_ids = batch["lang_ids"].to(device, non_blocking=True)
        
        # --- DANN Alpha Schedule ---
        # Update alpha only on optimization steps to be mathematically consistent
        # p goes from 0 to 1 over the course of training
        p = float(current_opt_step + (step // accumulation_steps)) / total_opt_steps
        p = max(0.0, min(1.0, p)) # Clip 0-1
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        # --- Forward Pass (Mixed Precision) ---
        with autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(
                input_ids, attention_mask, 
                style_feats=style_feats,
                lang_ids=lang_ids, labels=labels, alpha=alpha
            )
            loss = loss / accumulation_steps

        # --- Backward ---
        scaler.scale(loss).backward()

        # --- Optimizer Step ---
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        # --- Metrics Logging ---
        current_loss = loss.item() * accumulation_steps
        running_loss += current_loss
        
        # Quick Accuracy Estimate (Avoid storing all preds to save RAM)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
        
        progress_bar.set_postfix({
            "Loss": f"{current_loss:.4f}", 
            "Acc": f"{correct_preds/total_preds:.3f}",
            "LR": f"{scheduler.get_last_lr()[0]:.1e}"
        })

    # Return summary metrics
    epoch_loss = running_loss / len_dataloader
    epoch_acc = correct_preds / total_preds
    
    return {"loss": epoch_loss, "accuracy": epoch_acc}

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
    # 2. SMART DATA LOADING (CORRETTO)
    # =========================================================================
    # Ora ritorna 3 valori e i DF sono già separati e puliti
    train_df, val_df, language_map = load_data_for_training(config)
    
    # Aggiorna config col numero reale di linguaggi trovati
    config["model"]["num_languages"] = len(language_map)
    logger.info(f"Num languages identified: {len(language_map)}")
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    
    # 1. Calcolo pesi classi (Solo sul Train!)
    raw_weights = get_class_weights(train_df, device)
    
    # 2. Safety Check
    if raw_weights is not None and raw_weights.size(0) != 4:
        logger.warning(f"Attenzione: get_class_weights ha trovato solo {raw_weights.size(0)} classi.")
        logger.warning("Forzo l'uso di pesi uniformi (None) per evitare il crash.")
        class_weights = None
    else:
        logger.info(f"Class Weights: {raw_weights.cpu().numpy()}")
        class_weights = raw_weights
    
    # Dataset
    train_dataset = CodeDataset(train_df, tokenizer, language_map, config["data"]["max_length"], augment=True)
    val_dataset = CodeDataset(val_df, tokenizer, language_map, config["data"]["max_length"], augment=False)

    # Dataloader
    train_dl = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], 
                         shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    val_dl = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], 
                       shuffle=False, num_workers=4, pin_memory=True)

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

    # Optimizer & Scheduler
    # Filter only parameters that require gradients (in case of LoRA/Freezing)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(config["training"]["learning_rate"]), weight_decay=0.01)
    
    scaler = GradScaler() # Default for 'cuda'
    
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

    logger.info("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, 
            epoch, num_epochs, acc_steps
        )
        
        # Validation completa
        val_metrics, _, _ = evaluate(model, val_dl, device, verbose=False)
        
        # Logging
        ConsoleUX.log_metrics(f"Ep{epoch+1}", val_metrics)
        experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
        experiment.log_metrics(val_metrics, prefix="Val", step=epoch)

        # Checkpointing
        current_f1 = val_metrics["f1"]
        if current_f1 > best_f1:
            logger.info(f"🚀 New Best F1: {current_f1:.4f} (was {best_f1:.4f})")
            best_f1 = current_f1
            patience_counter = 0
            
            # Save logic handling LoRA vs Full Model
            save_path = os.path.join(checkpoint_dir, "best_model_state.pt")
            
            if model.use_lora:
                # Save adapter + custom heads
                model.base_model.save_pretrained(checkpoint_dir) # Saves LoRA adapter
                
                # Save custom heads manually
                heads_state = {
                    'classifier': model.classifier.state_dict(),
                    'language_classifier': model.language_classifier.state_dict(),
                    'pooler': model.pooler.state_dict(),
                    'projection': model.projection_head.state_dict(),
                    'norms': {
                        'semantic': model.norm_semantic.state_dict(),
                        'style': model.norm_style.state_dict()
                    }
                }
                torch.save(heads_state, os.path.join(checkpoint_dir, "custom_heads.pt"))
            else:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    experiment.end()
    logger.info("Training Finished.")