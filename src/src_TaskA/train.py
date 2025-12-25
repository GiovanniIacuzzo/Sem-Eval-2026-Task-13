import os
import logging
import yaml
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler 
from torch.optim.swa_utils import AveragedModel, SWALR
from dotenv import load_dotenv
from comet_ml import Experiment

from src.src_TaskA.models.model import FusionCodeClassifier 
from src.src_TaskA.dataset.dataset import load_data
from src.src_TaskA.utils.utils import evaluate

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

class ConsoleUX:
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")
    @staticmethod
    def log_metrics(stage, metrics):
        log = f"[{stage}] "
        for k, v in metrics.items():
            val = f"{v:.4f}" if isinstance(v, float) else f"{v}"
            log += f"{k}: {val} | "
        logger.info(log.strip(" | "))

# -----------------------------------------------------------------------------
# Optimizer Params
# -----------------------------------------------------------------------------
def get_llrd_optimizer_params(model, base_lr, weight_decay=0.01, decay_factor=0.95):
    opt_parameters = []
    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    head_names = [n for n, p in named_parameters if "base_model" not in n]
    
    # Head
    opt_parameters.append({
        "params": [p for n, p in named_parameters if n in head_names and not any(nd in n for nd in no_decay)],
        "lr": base_lr, "weight_decay": weight_decay
    })
    opt_parameters.append({
        "params": [p for n, p in named_parameters if n in head_names and any(nd in n for nd in no_decay)],
        "lr": base_lr, "weight_decay": 0.0
    })
    
    # Backbone
    if hasattr(model, "base_model") and hasattr(model.base_model, "encoder"):
        layers = list(model.base_model.encoder.layer)
        layers.reverse()
        lr = base_lr
        for layer in layers:
            lr *= decay_factor
            decay = [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)]
            nodecay = [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)]
            opt_parameters.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
            opt_parameters.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})
        
        emb_params = list(model.base_model.embeddings.parameters())
        lr *= decay_factor
        opt_parameters.append({"params": emb_params, "lr": lr, "weight_decay": weight_decay})
    else:
        # Fallback
        backbone = [p for n, p in named_parameters if "base_model" in n]
        opt_parameters.append({"params": backbone, "lr": base_lr * 0.8, "weight_decay": weight_decay})
        
    return opt_parameters

# -----------------------------------------------------------------------------
# Train Epoch
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, 
                   epoch_idx, accumulation_steps=1, swa_model=None, swa_scheduler=None):
    
    model.train()
    running_loss = 0.0
    ce_criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}", leave=False)
    
    for step, batch in enumerate(pbar):
        # Sposta tutto su GPU
        inputs = {k: v.to(device) for k, v in batch.items()}
        
        with autocast('cuda', dtype=torch.float16):
            # Forward
            logits, loss = model(
                inputs["input_ids"], 
                inputs["attention_mask"], 
                inputs.get("stylo_feats"), 
                labels=inputs["labels"]
            )
            
            if loss is None:
                loss = ce_criterion(logits, inputs["labels"])

            loss = loss / accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if swa_model:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            elif scheduler:
                scheduler.step()
        
        running_loss += loss.item() * accumulation_steps
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return {"loss": running_loss / len(dataloader)}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    ConsoleUX.print_banner("SemEval 2026 - Task 13 A")
    
    # 1. Load Config
    with open("src/src_TaskA/config/config.yaml") as f: config = yaml.safe_load(f)
    
    # Auto-find paths
    base_dir = "data/Task_A"
    if not config["data"].get("train_path"):
        config["data"]["train_path"] = os.path.join(base_dir, "train.parquet")
    if not config["data"].get("val_path"):
        config["data"]["val_path"] = os.path.join(base_dir, "validation.parquet")

    # 2. Setup Device & Logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Comet
    exp = None
    if os.getenv("COMET_API_KEY"):
        exp = Experiment(api_key=os.getenv("COMET_API_KEY"), project_name=os.getenv("COMET_PROJECT_NAME"))
        exp.log_parameters(config)

    # 3. Model & Data
    model = FusionCodeClassifier(config).to(device)
    train_ds, val_ds, train_sampler, _ = load_data(config, model.tokenizer)

    # DataLoader
    train_dl = DataLoader(
        train_ds, batch_size=config["training"]["batch_size"], 
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True
    )
    
    val_bs = config["training"]["batch_size"] * 2
    val_dl = DataLoader(
        val_ds, batch_size=val_bs, shuffle=False, 
        num_workers=4, pin_memory=True
    )

    # 4. Training Setup
    optimizer = torch.optim.AdamW(get_llrd_optimizer_params(model, float(config["training"]["learning_rate"])))
    scaler = GradScaler()
    
    num_epochs = config["training"]["num_epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(config["training"]["learning_rate"]),
        total_steps=len(train_dl)*num_epochs, pct_start=0.1
    )
    
    # SWA
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=2e-6)
    swa_start = int(num_epochs * 0.75)

    # 5. Loop
    best_f1 = 0.0
    save_path = config["training"]["checkpoint_dir"]
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        use_swa = (epoch >= swa_start)
        
        metrics = train_one_epoch(
            model, train_dl, optimizer, 
            None if use_swa else scheduler, scaler, device, epoch, 
            accumulation_steps=config["training"]["gradient_accumulation_steps"],
            swa_model=swa_model if use_swa else None,
            swa_scheduler=swa_scheduler if use_swa else None
        )
        ConsoleUX.log_metrics("Train", metrics)
        
        # Validation
        eval_net = swa_model if use_swa else model
        if use_swa: torch.optim.swa_utils.update_bn(train_dl, swa_model, device=device)
        
        val_metrics, _, _ = evaluate(eval_net, val_dl, device)
        ConsoleUX.log_metrics("Valid", val_metrics)
        
        if exp:
            exp.log_metrics(metrics, prefix="Train", step=epoch)
            exp.log_metrics(val_metrics, prefix="Val", step=epoch)
            
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pt"))
            logger.info(f"Saved Best Model (F1: {best_f1:.4f})")

    logger.info("Done.")