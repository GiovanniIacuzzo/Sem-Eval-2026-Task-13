import os
import sys
import logging
import yaml
import torch
import random
import numpy as np
import gc
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from comet_ml import Experiment

from src.src_TaskA.models.model import FusionCodeClassifier
from src.src_TaskA.dataset.dataset import CodeDataset, BalancedBatchSampler, load_full_data_for_kfold
from src.src_TaskA.utils.utils import evaluate

# -----------------------------------------------------------------------------
# Setup & Config
# -----------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConsoleUX:
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

def seed_everything(seed=42):
    """Garantisce la riproducibilità totale."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Training Loop Singola Epoca
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch_idx, accum_steps=1):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Ep {epoch_idx+1}", leave=False, dynamic_ncols=True)
    
    # Reset gradienti all'inizio
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        stylo_feats = batch.get("stylo_feats")
        labels = batch["labels"].to(device, non_blocking=True)
        
        if stylo_feats is not None:
            stylo_feats = stylo_feats.to(device, non_blocking=True)

        with autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(
                input_ids, attention_mask, stylo_feats=stylo_feats, labels=labels
            )
            # Normalizza la loss per l'accumulo gradienti
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        # Step dell'ottimizzatore solo ogni 'accum_steps'
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if scheduler:
                scheduler.step()
        
        # Riporta la loss alla scala originale per il logging
        running_loss += loss.item() * accum_steps
        pbar.set_postfix({"Loss": f"{running_loss / (step + 1):.4f}"})

    return running_loss / len(dataloader)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    ConsoleUX.print_banner("Task A - K-Fold Fusion Training (Clean)")
    
    # 1. Config & Seed
    with open("src/src_TaskA/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    SEED = config.get("training", {}).get("seed", 42)
    seed_everything(SEED)
    
    DATA_ROOT = os.getenv("DATA_PATH", "./data/Task_A")
    config["data"]["train_path"] = os.path.join(DATA_ROOT, "train.parquet")
    config["data"]["val_path"] = os.path.join(DATA_ROOT, "validation.parquet")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")
    
    # 2. Caricamento Dati Unificato
    full_df = load_full_data_for_kfold(config)
    y_all = full_df['label'].values
    
    # Mappa lingue dinamica
    unique_langs = sorted(full_df['language'].unique().astype(str))
    language_map = {l: i for i, l in enumerate(unique_langs)}
    if 'unknown' not in language_map: language_map['unknown'] = len(language_map)
    logger.info(f"Language Map Created: {len(language_map)} languages.")

    # Tokenizer
    model_name = config["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. K-Fold Setup
    k_folds = 5
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    
    # Setup Comet
    experiment = None
    if os.getenv("COMET_API_KEY"):
        try:
            experiment = Experiment(
                api_key=os.getenv("COMET_API_KEY"),
                project_name="semeval-task-a-fusion-clean",
                auto_metric_logging=False
            )
            experiment.log_parameters(config)
        except Exception as e:
            logger.warning(f"Comet init failed: {e}")

    # -------------------------------------------------------------------------
    # LOOP DEI FOLD
    # -------------------------------------------------------------------------
    results_per_fold = []
    
    # Array per salvare le previsioni Out-Of-Fold
    oof_preds = np.zeros((len(full_df), 2))
    oof_targets = np.zeros(len(full_df))

    grad_accum_steps = config["training"].get("grad_accum_steps", 1)

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(y_all)), y_all)):
        ConsoleUX.print_banner(f"FOLD {fold+1}/{k_folds}")
        
        # Split Dataframe
        train_df_fold = full_df.iloc[train_idx].reset_index(drop=True)
        val_df_fold = full_df.iloc[val_idx].reset_index(drop=True)
        
        train_dataset = CodeDataset(train_df_fold, tokenizer, language_map, 
                                   max_length=config["data"]["max_length"])
        val_dataset = CodeDataset(val_df_fold, tokenizer, language_map, 
                                 max_length=config["data"]["max_length"])
        
        train_sampler = BalancedBatchSampler(
            train_dataset.labels_list, 
            batch_size=config["training"]["batch_size"]
        )
        
        train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, 
                             num_workers=4, pin_memory=True)
    
        val_dl = DataLoader(val_dataset, batch_size=config["training"]["batch_size"]*2, 
                           shuffle=False, num_workers=4, pin_memory=True)
        
        logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

        # Init Model
        model = FusionCodeClassifier(config)
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=0.01)
        scaler = GradScaler()
        
        num_epochs = config["training"]["num_epochs"]
        
        steps_per_epoch = len(train_dl) // grad_accum_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=float(config["training"]["learning_rate"]),
            total_steps=steps_per_epoch * num_epochs, 
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1000.0
        )

        best_f1 = 0.0
        fold_save_path = os.path.join(config["training"]["checkpoint_dir"], f"fold_{fold}")
        os.makedirs(fold_save_path, exist_ok=True)
        
        # Variabili per OOF del fold corrente
        best_val_preds = None

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(
                model, train_dl, optimizer, scheduler, scaler, device, epoch, 
                accum_steps=grad_accum_steps
            )
            
            # Valutazione
            val_metrics, val_preds, val_targets = evaluate(model, val_dl, device)
            
            logger.info(f"Ep {epoch+1} | Loss: {train_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
            
            if experiment:
                experiment.log_metrics(val_metrics, prefix=f"Fold{fold}_Val", step=(fold * num_epochs) + epoch)
            
            # Save best & Store OOF
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save(model.state_dict(), os.path.join(fold_save_path, "best_model.pt"))
                
                # Salviamo le predizioni del miglior modello per questo fold
                oof_preds[val_idx] = np.array(val_preds).reshape(-1, 1)
                oof_targets[val_idx] = np.array(val_targets)
                logger.info("--> New Best Model Saved")

        results_per_fold.append(best_f1)
        
        # Cleanup
        del model, optimizer, scheduler, scaler, train_dl, val_dl
        torch.cuda.empty_cache()
        gc.collect()

    # -------------------------------------------------------------------------
    # RISULTATI FINALI
    # -------------------------------------------------------------------------
    avg_f1 = np.mean(results_per_fold)
    ConsoleUX.print_banner(f"TRAINING COMPLETED")
    logger.info(f"Average F1 across {k_folds} folds: {avg_f1:.4f}")
    
    # Calcolo Score su tutto il dataset unito
    oof_flat_preds = oof_preds[:, 0]
    mask = oof_targets != 0
    
    # Calcolo metriche globali
    report = classification_report(oof_targets, oof_flat_preds, target_names=['Human', 'Machine'])
    print("\nGlobal OOF Classification Report:\n")
    print(report)

    if experiment:
        experiment.log_metric("CV_Average_F1", avg_f1)