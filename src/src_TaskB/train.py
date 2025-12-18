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

# Imports locali
from src.src_TaskB.models.model import CodeClassifier
from src.src_TaskB.dataset.dataset import load_data
from src.src_TaskB.utils.utils import evaluate

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

# Definiamo le etichette per le due modalità
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
        # Priorità: F1 Macro > Accuracy > Loss
        keys = ["f1_macro", "f1_weighted", "accuracy", "loss"] + [k for k in metrics.keys() if k not in ["f1_macro", "f1_weighted", "accuracy", "loss"]]
        
        for k in keys:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, float):
                    log_str += f"{k}: {v:.4f} | "
                else:
                    log_str += f"{k}: {v} | "
        logger.info(log_str.strip(" | "))

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False

def save_checkpoint(model, path, is_peft=False):
    """
    Salva il modello gestendo correttamente le parti PEFT e Custom.
    """
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving model to {path}...")
    
    if is_peft:
        # 1. Salva gli adapter LoRA
        model.base_model.save_pretrained(path)
        
        # 2. Salva le componenti Custom (Classifier, Heads)
        custom_state = {
            'classifier': model.classifier.state_dict(),
            'pooler': model.pooler.state_dict(),
            'projection_head': model.projection_head.state_dict(),
            'language_classifier': model.language_classifier.state_dict()
        }
        torch.save(custom_state, os.path.join(path, "custom_components.pt"))
    else:
        # Full model save
        torch.save(model.state_dict(), os.path.join(path, "full_model.bin"))
        # Salviamo anche la config del tokenizer per comodità
        model.tokenizer.save_pretrained(path)

# -----------------------------------------------------------------------------
# Training Routine
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, 
                   epoch_idx, total_epochs, accumulation_steps=1):
    
    model.train()
    running_loss = 0.0
    predictions, references = [], []
    
    optimizer.zero_grad()
    
    # Progress Bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}", leave=False, dynamic_ncols=True)
    len_dataloader = len(dataloader)
    
    for step, batch in enumerate(progress_bar):
        # Spostamento su GPU
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        # Gestione opzionale lang_ids (potrebbe non esserci nel binary)
        lang_ids = batch.get("lang_ids", None)
        if lang_ids is not None:
            lang_ids = lang_ids.to(device, non_blocking=True)
        
        # --- DANN Alpha Scheduling ---
        # Aumenta l'intensità dell'adversarial training col tempo
        current_step = step + epoch_idx * len_dataloader
        total_steps = total_epochs * len_dataloader
        p = float(current_step) / (total_steps + 1e-8)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        # Mixed Precision Context
        with autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(
                input_ids, 
                attention_mask, 
                lang_ids=lang_ids, 
                labels=labels, 
                alpha=alpha
            )
            loss = loss / accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        # Optimizer Step (Gradient Accumulation)
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        # Logging immediato
        current_loss = loss.item() * accumulation_steps
        running_loss += current_loss
        
        progress_bar.set_postfix({
            "Loss": f"{current_loss:.4f}", 
            "Alpha": f"{alpha:.2f}", 
            "LR": f"{scheduler.get_last_lr()[0]:.1e}"
        })

        # Raccolta metriche
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        predictions.extend(preds)
        references.extend(labels_cpu)

    # Calcolo metriche fine epoca
    metrics = model.compute_metrics(predictions, references)
    metrics["loss"] = running_loss / len_dataloader
    
    return metrics

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src_TaskB/config/config.yaml")
    parser.add_argument("--mode", type=str, required=True, choices=["binary", "families"], 
                        help="Choose training mode: 'binary' (Human vs AI) or 'families' (Specific AI attribution)")
    args = parser.parse_args()
    
    ConsoleUX.print_banner(f"SemEval 2026 Task 13 - Mode: {args.mode.upper()}")

    # 1. Load & Merge Config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)

    # MERGE CONFIGURATION
    # Prendiamo 'common' e sovrascriviamo con la sezione specifica (binary o families)
    mode_config = raw_config["common"].copy()
    if args.mode in raw_config:
        mode_config.update(raw_config[args.mode])
    else:
        logger.error(f"Config section for '{args.mode}' not found in yaml!")
        sys.exit(1)
        
    # Ricostruiamo la struttura attesa dal Modello (nidificata)
    # Il CodeClassifier si aspetta config["model"]...
    final_config = {
        "model": {
            "model_name": mode_config["model_name"],
            "num_labels": mode_config["num_labels"],
            "use_lora": mode_config.get("use_lora", False),
            "lora_r": mode_config.get("lora_r", 32),
            "languages": mode_config.get("languages", []),
            "class_weights": mode_config.get("class_weights", False) # Flag per logica interna
        },
        "training": mode_config,
        "data": mode_config
    }
    
    # Label Names dinamici per i log
    current_label_names = LABELS_BINARY if args.mode == "binary" else LABELS_FAMILIES

    # 2. Experiment Tracking
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME"),
        workspace=os.getenv("COMET_WORKSPACE"),
        auto_metric_logging=False
    )
    experiment.add_tag(args.mode)
    experiment.log_parameters(mode_config)

    # 3. Device Setup
    if not torch.cuda.is_available():
        logger.warning("CUDA not found! Training will be slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 4. Tokenizer Init
    model_name = mode_config["model_name"]
    logger.info(f"Loading Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 5. Data Loading
    # Passiamo 'args.mode' a load_data per caricare il parquet giusto
    logger.info(f"Loading Data for mode: {args.mode}...")
    train_dataset, val_dataset, class_weights = load_data(final_config, tokenizer, mode=args.mode)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info(f"Class Weights loaded: {class_weights}")
    else:
        logger.info("No Class Weights used (Balanced Dataset).")

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

    # 6. Model Init
    logger.info(f"Initializing Model ({args.mode})...")
    # Passiamo la final_config ricostruita
    model_wrapper = CodeClassifier(final_config, class_weights=class_weights)
    model_wrapper.to(device)

    is_peft = hasattr(model_wrapper, 'use_lora') and model_wrapper.use_lora
    if is_peft:
        logger.info(f"LoRA Active. Trainable params: {sum(p.numel() for p in model_wrapper.parameters() if p.requires_grad)}")

    # 7. Optimization
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

    # 8. Training Loop
    best_f1 = float("-inf")
    patience = 4 # Hardcoded patience is usually fine
    patience_counter = 0

    # Cartella di output specifica per la modalità
    checkpoint_dir = os.path.join(mode_config["checkpoint_dir"], args.mode)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        ConsoleUX.print_banner(f"Epoch {epoch+1}/{num_epochs}")

        # Train
        train_metrics = train_one_epoch(
            model_wrapper, train_dl, optimizer, scheduler, scaler, device, 
            epoch, num_epochs, acc_steps
        )
        ConsoleUX.log_metrics("Train", train_metrics)
        
        # Validate
        torch.cuda.empty_cache() 
        val_metrics, val_preds, val_refs = evaluate(model_wrapper, val_dl, device)
        ConsoleUX.log_metrics("Valid", val_metrics)

        # Logging Comet
        experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
        experiment.log_metrics(val_metrics, prefix="Val", step=epoch)

        # Checkpointing
        current_f1 = val_metrics.get("f1_macro", val_metrics.get("f1", 0.0))
        
        if current_f1 > best_f1:
            logger.info(f"🚀 New Best Macro F1: {current_f1:.4f} (was {best_f1:.4f})")
            best_f1 = current_f1
            patience_counter = 0
            
            save_checkpoint(model_wrapper, checkpoint_dir, is_peft=is_peft)
            
            # Confusion Matrix
            try:
                cm = confusion_matrix(val_refs, val_preds)
                # Verifica che il numero di label corrisponda
                if len(cm) == len(current_label_names):
                    experiment.log_confusion_matrix(
                        matrix=cm, 
                        title=f"CM_{args.mode}_Epoch_{epoch}",
                        labels=current_label_names,
                        file_name=f"confusion_matrix_epoch_{epoch}.json"
                    )
            except Exception as e:
                logger.warning(f"Could not log confusion matrix: {e}")
                
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    experiment.end()
    logger.info(f"Training Complete for mode: {args.mode}")