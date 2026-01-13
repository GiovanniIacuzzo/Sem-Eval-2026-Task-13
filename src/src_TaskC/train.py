import os
import sys
import logging
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer
from comet_ml import Experiment
from pytorch_metric_learning import losses

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from src.src_TaskC.models.model import CodeClassifier
from src.src_TaskC.dataset.dataset import CodeDataset, load_data_for_training, get_class_weights
from src.src_TaskC.utils.utils import evaluate, set_seed

# -----------------------------------------------------------------------------
# Logger & UX Setup
# -----------------------------------------------------------------------------
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
        keys = sorted(metrics.keys(), key=lambda x: (0 if 'f1' in x else 1 if 'acc' in x else 2, x))
        log_str = f"[{stage}] "
        for k in keys:
            v = metrics[k]
            if "class" in k: 
                short_k = k.replace('f1_class_', 'C')
                log_str += f"{short_k}: {v:.3f} | "
            elif isinstance(v, float):
                log_str += f"{k}: {v:.4f} | "
            else:
                log_str += f"{k}: {v} | "
        logger.info(log_str.strip(" | "))

# -----------------------------------------------------------------------------
# Training Routine
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, 
                   epoch_idx, accumulation_steps=1, contrastive_fn=None):
    
    model.train()
    
    running_metrics = {"loss": 0.0, "task_loss": 0.0, "con_loss": 0.0}
    correct_preds = 0
    total_preds = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    progress_bar = tqdm(dataloader, desc=f"Ep {epoch_idx+1} Train", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        extra_features = batch["extra_features"].to(device, non_blocking=True)

        with autocast(device_type='cuda', dtype=torch.float16):
            # Model Forward
            logits, loss_task, proj_features = model(
                input_ids, attention_mask, 
                labels=labels, 
                extra_features=extra_features
            )
            
            # Contrastive Loss (SupCon)
            loss_con = torch.tensor(0.0, device=device)
            if contrastive_fn is not None:
                loss_con = contrastive_fn(proj_features, labels)

            alpha = 0.5
            total_loss = (loss_task + alpha * loss_con) / accumulation_steps

        scaler.scale(total_loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if scheduler is not None:
                scheduler.step()
        
        current_loss = total_loss.item() * accumulation_steps
        running_metrics["loss"] += current_loss
        running_metrics["task_loss"] += loss_task.item()
        running_metrics["con_loss"] += loss_con.item()
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
        
        progress_bar.set_postfix({
            "L": f"{current_loss:.3f}", 
            "T": f"{loss_task.item():.3f}",
            "C": f"{loss_con.item():.3f}",
            "Acc": f"{correct_preds/total_preds:.2f}"
        })

    avg_loss = running_metrics["loss"] / len(dataloader)
    epoch_acc = correct_preds / total_preds
    
    return {
        "loss": avg_loss, 
        "accuracy": epoch_acc,
        "task_loss": running_metrics["task_loss"] / len(dataloader),
        "con_loss": running_metrics["con_loss"] / len(dataloader)
    }

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/src_TaskC/config/config.yaml") 
    parser.add_argument("--stage", type=str, default="detection", 
                        choices=["detection", "attribution_obf", "end2end"],
                        help="detection: Clean(0) vs Obf(1) | attribution_obf: Human-Obf(0) vs AI-Obf(1) | end2end: 4 classes")
    args = parser.parse_args()

    ConsoleUX.print_banner(f"SemEval 2026 - Task 13 C - [{args.stage.upper()}]")

    # --- 1. SETUP CONFIG & DEVICE ---
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # =========================================================================
    # 2. DATA PREPARATION
    # =========================================================================
    train_df, val_df = load_data_for_training(config)
    
    if args.stage == "detection":
        logger.info("Applying Label Mapping for DETECTION Stage (Clean=0, Obf=1)...")
        
        train_df['label'] = train_df['label'].apply(lambda x: 1 if x in [2, 3] else 0)
        val_df['label'] = val_df['label'].apply(lambda x: 1 if x in [2, 3] else 0)
        
        num_labels = 2
        checkpoint_dir = os.path.join(config["training"]["checkpoint_dir"], "stage1_detection")

    elif args.stage == "attribution_obf":
        logger.info("Filtering Data for ATTRIBUTION_OBF Stage (Human-Obf vs AI-Obf)...")
        
        train_df = train_df[train_df['label'].isin([2, 3])].copy()
        val_df = val_df[val_df['label'].isin([2, 3])].copy()
        
        label_map = {2: 0, 3: 1}
        train_df['label'] = train_df['label'].map(label_map)
        val_df['label'] = val_df['label'].map(label_map)
        
        num_labels = 2
        checkpoint_dir = os.path.join(config["training"]["checkpoint_dir"], "stage2_attribution_obf")
        
    else:
        logger.info("Training in END-TO-END mode (4 Classes)...")
        num_labels = 4
        checkpoint_dir = os.path.join(config["training"]["checkpoint_dir"], "end2end")

    logger.info(f"Training Samples: {len(train_df)} | Validation Samples: {len(val_df)}")
    logger.info(f"Class Distribution Train: {train_df['label'].value_counts().to_dict()}")

    config["model"]["num_labels"] = num_labels
    os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    
    class_weights = get_class_weights(train_df, device)
    
    # Dataset
    train_dataset = CodeDataset(train_df, tokenizer, max_length=config["data"]["max_length"], is_train=True)
    val_dataset = CodeDataset(val_df, tokenizer, max_length=config["data"]["max_length"], is_train=False)

    # Dataloaders
    train_dl = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=config.get("num_workers", 4), 
        pin_memory=True, 
        drop_last=True
    )
    val_dl = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=config.get("num_workers", 4), 
        pin_memory=True
    )

    # =========================================================================
    # 3. MODEL & EXPERIMENT SETUP
    # =========================================================================
    model = CodeClassifier(config, class_weights=class_weights)
    model.to(device)

    # Comet ML
    api_key = os.getenv("COMET_API_KEY")
    experiment = None
    if api_key:
        experiment = Experiment(
            api_key=api_key,
            project_name=os.getenv("COMET_PROJECT_NAME", "semeval-task13-subtaskc"),
            workspace=os.getenv("COMET_WORKSPACE"),
            auto_metric_logging=False
        )
        experiment.log_parameters(config)
        experiment.add_tag(args.stage)

    # Optimizer
    lr = float(config["training"]["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=config["training"].get("weight_decay", 0.01)
    )
    
    scaler = GradScaler()
    
    acc_steps = config["training"].get("gradient_accumulation_steps", 1)
    num_epochs = config["training"].get("num_epochs", 5) 
    total_steps = (len(train_dl) // acc_steps) * num_epochs
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1
    )

    contrastive_fn = losses.SupConLoss(temperature=0.07).to(device)

    # =========================================================================
    # 4. TRAINING LOOP
    # =========================================================================
    best_f1 = 0.0
    patience = config["training"].get("early_stop_patience", 4)
    patience_counter = 0

    logger.info(f"Starting Training for STAGE: {args.stage}...")
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, 
            epoch, accumulation_steps=acc_steps, contrastive_fn=contrastive_fn
        )
        
        # Valid
        val_metrics, _, _ = evaluate(model, val_dl, device, verbose=False)
        
        # Log
        ConsoleUX.log_metrics(f"Ep{epoch+1}", val_metrics)
        if experiment:
            experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
            experiment.log_metrics(val_metrics, prefix="Val", step=epoch)
            experiment.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

        # Checkpoint
        current_f1 = val_metrics.get("f1_macro", val_metrics.get("f1", 0.0))
        
        if current_f1 > best_f1:
            logger.info(f"---> New Best F1: {current_f1:.4f} (prev: {best_f1:.4f})")
            best_f1 = current_f1
            patience_counter = 0
            
            save_path = os.path.join(checkpoint_dir, "best_model.bin")
            torch.save(model.state_dict(), save_path)
            
            with open(os.path.join(checkpoint_dir, "model_meta.yaml"), "w") as f:
                yaml.dump({
                    "stage": args.stage,
                    "num_labels": num_labels,
                    "best_f1": best_f1,
                    "epoch": epoch
                }, f)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            ConsoleUX.print_banner("EARLY STOPPING TRIGGERED")
            break

    if experiment:
        experiment.end()
    
    logger.info(f"Training for {args.stage} Finished. Model saved in {checkpoint_dir}")