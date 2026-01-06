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
from peft import PeftModel

from torch.nn.utils.rnn import pad_sequence

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

class DynamicCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch])
        extra_features = torch.stack([item['extra_features'] for item in batch])
        
        # Padding dinamico solo fino alla lunghezza massima del batch corrente
        padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        return {
            "input_ids": padded_ids,
            "attention_mask": padded_mask,
            "labels": labels,
            "extra_features": extra_features
        }

def save_checkpoint(model, path, tokenizer, epoch, f1_score):
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving model to {path} (Epoch {epoch}, F1: {f1_score:.4f})...")
    
    tokenizer.save_pretrained(path)
    model.base_model.save_pretrained(path) # Salva LoRA adapter
    
    # Salviamo le teste custom separatamente
    torch.save(model.classifier.state_dict(), os.path.join(path, "classifier_head.pt"))
    torch.save(model.extra_projector.state_dict(), os.path.join(path, "projector.pt"))
    
    # Salviamo un file di metadati semplice
    with open(os.path.join(path, "metadata.txt"), "w") as f:
        f.write(f"epoch: {epoch}\n")
        f.write(f"f1_score: {f1_score}\n")

def load_checkpoint_for_resume(model, checkpoint_path, device):
    """Carica i pesi per riprendere il training in modo robusto."""
    logger.info(f"RESUMING TRAINING from {checkpoint_path}...")
    
    # 1. Load LoRA Adapter
    try:
        model.base_model.load_adapter(checkpoint_path, adapter_name="default")
        logger.info("LoRA adapters loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load LoRA with load_adapter ({e}). Trying PeftModel direct load...")
        model.base_model = PeftModel.from_pretrained(model.base_model.base_model, checkpoint_path)

    # 2. Load Custom Heads
    head_path = os.path.join(checkpoint_path, "classifier_head.pt")
    proj_path = os.path.join(checkpoint_path, "projector.pt")
    
    # --- CLASSIFIER HEAD ---
    if os.path.exists(head_path):
        try:
            state_dict = torch.load(head_path, map_location=device, weights_only=True)
            model.classifier.load_state_dict(state_dict)
            logger.info("Classifier head loaded successfully.")
        except RuntimeError as e:
            logger.warning(f"ARCHITECTURE MISMATCH for Classifier Head: {e}")
            logger.warning("--> Discarding old head weights and initializing from scratch (Recommended for architectural changes).")
    else:
        logger.warning(f"Head weights not found at {head_path}! Initializing from scratch.")
    
    # --- PROJECTOR ---
    if os.path.exists(proj_path):
        try:
            state_dict = torch.load(proj_path, map_location=device, weights_only=True)
            model.extra_projector.load_state_dict(state_dict)
            logger.info("Projector loaded successfully.")
        except RuntimeError as e:
            logger.warning(f"ARCHITECTURE MISMATCH for Projector: {e}")
            logger.warning("--> Discarding old projector weights and initializing from scratch.")
    else:
        logger.warning(f"Projector weights not found at {proj_path}! Initializing from scratch.")
        
    return model

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, 
                   epoch_idx, total_epochs, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    
    optimizer.zero_grad(set_to_none=True)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}/{total_epochs}", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        # Gestione extra features
        extra_features = batch.get("extra_features", None)
        if extra_features is not None:
            extra_features = extra_features.to(device, non_blocking=True, dtype=torch.bfloat16)
        
        dtype_amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Mixed Precision Training
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
            
            optimizer.zero_grad(set_to_none=True)
            
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
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to start from (useful if resuming)")
    args = parser.parse_args()
    
    ConsoleUX.print_banner("SemEval 2026 - Task 13 - Subtask A")
    set_seed(42)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)["common"]

    # Inizializzazione Comet ML
    api_key = os.getenv("COMET_API_KEY")
    if api_key:
        experiment = Experiment(
            api_key=api_key,
            project_name=os.getenv("COMET_PROJECT_NAME", "semeval-task-a"),
            auto_metric_logging=False
        )
        experiment.set_name(f"StarCoder_{config['model_name'].split('/')[-1]}_Resume" if args.resume_from else f"StarCoder_Run")
        experiment.log_parameters(config)
    else:
        logger.warning("COMET_API_KEY not found. Logging will be local only.")
        experiment = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")

    # --- 1. Tokenizer & Data ---
    model_path_for_tokenizer = args.resume_from if args.resume_from else config["model_name"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path_for_tokenizer, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load tokenizer from checkpoint ({e}). Loading base tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    train_dataset, val_dataset, class_weights = load_data(config, tokenizer)
    label_names = ["Human", "AI"]
    
    if class_weights is not None:
        class_weights = class_weights.to(device)

    collate_fn = DynamicCollate(tokenizer)

    # Ottimizzazione DataLoader per L4
    train_dl = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_dl = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    # --- 2. Model Setup ---
    model = CodeModel(config, class_weights=class_weights)
    
    # Logica per gestire il Learning Rate e lo Scheduler in caso di Resume
    base_lr = float(config["learning_rate"])
    acc_steps = config.get("gradient_accumulation_steps", 1)
    
    if args.resume_from:
        # Carica pesi
        model = load_checkpoint_for_resume(model, args.resume_from, "cpu")
        model.to(device)
        
        start_epoch = args.start_epoch
        if start_epoch == 0:
            logger.warning("Resuming but start_epoch is 0. If you already trained 2 epochs, pass --start_epoch 2")
        
        # STRATEGIA RESUME: LR ridotto e Cosine Decay senza Warmup
        current_lr = base_lr / 4.0
        logger.info(f"Resume Strategy: Starting with reduced LR {current_lr}")
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr)
        
        steps_remaining = len(train_dl) * (config["num_epochs"] - start_epoch) // acc_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=steps_remaining, 
            eta_min=1e-6
        )
    else:
        model.to(device)
        start_epoch = 0
        current_lr = base_lr
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr)
        
        total_steps = len(train_dl) * config["num_epochs"] // acc_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=current_lr, 
            total_steps=total_steps,
            pct_start=0.1 # 10% warmup
        )

    scaler = GradScaler()
    
    best_f1 = 0.0
    patience_counter = 0 
    patience = config.get("early_stop_patience", 3)
    checkpoint_dir = config["checkpoint_dir"]

    # --- 3. Training Loop ---
    logger.info(f"Starting training from epoch {start_epoch+1}")
    
    for epoch in range(start_epoch, config["num_epochs"]):
        ConsoleUX.print_banner(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        train_metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, scaler, device, 
            epoch, config["num_epochs"], acc_steps
        )
        
        if experiment:
            experiment.log_metrics(train_metrics, prefix="Train", step=epoch)
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Validation
        val_metrics, val_preds, val_refs, val_report = evaluate_model(model, val_dl, device, label_names=label_names)
        
        ConsoleUX.log_metrics("Valid", val_metrics)
        if experiment:
            experiment.log_metrics(val_metrics, prefix="Val", step=epoch)
        logger.info(f"\n{val_report}")

        current_f1 = val_metrics["f1_macro"]
        
        # Logica Salvataggio
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            
            save_path = os.path.join(checkpoint_dir, "best_model_resumed" if args.resume_from else "best_model")
            save_checkpoint(model, save_path, tokenizer, epoch+1, best_f1)
            
            if experiment:
                cm = confusion_matrix(val_refs, val_preds)
                experiment.log_confusion_matrix(matrix=cm, title=f"Confusion Matrix Epoch {epoch}", labels=label_names)
            logger.info(f"--> New Best F1: {best_f1:.4f}. Model Saved.")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.warning("Early Stopping Triggered")
                break
    
    if experiment:
        experiment.end()

if __name__ == "__main__":
    main()