import os
import sys
import logging
import yaml
import torch
import numpy as np
import zipfile
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.amp import autocast
from dotenv import load_dotenv
from copy import deepcopy
from sklearn.metrics import confusion_matrix, classification_report

from comet_ml import Experiment

# Assicurati di aver installato la lib: pip install kaggle
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Errore: Libreria 'kaggle' non trovata. Installala con 'pip install kaggle'")
    sys.exit(1)

from src_TaskA.models.model import CodeClassifier
from src_TaskA.dataset.dataset import load_data
from src_TaskA.utils.utils import evaluate

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConsoleUX:
    """Helper class for cleaner console output."""
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(stage, metrics):
        log_str = f"[{stage}] "
        for k, v in metrics.items():
            if isinstance(v, float):
                log_str += f"{k}: {v:.4f} | "
            else:
                log_str += f"{k}: {v} | "
        logger.info(log_str.strip(" | "))

# -----------------------------------------------------------------------------
# Kaggle Data Download Helper
# -----------------------------------------------------------------------------
def setup_kaggle_data(download_dir="./data"):
    """
    Scarica e scompatta i dati della competizione se non sono presenti.
    Richiede KAGGLE_USERNAME e KAGGLE_KEY nel .env o nell'ambiente.
    """
    # Nome della competizione (Subtask A)
    # NOTA: Verifica che questo slug sia corretto controllando l'URL della competizione su Kaggle
    COMPETITION_SLUG = "sem-eval-2026-task-13-subtask-a" 
    
    task_dir = os.path.join(download_dir, "Task_A")
    train_file = os.path.join(task_dir, "train.parquet")
    
    # Se il file esiste già, saltiamo il download
    if os.path.exists(train_file):
        logger.info(f"Dati trovati in {task_dir}. Skip download.")
        return task_dir

    logger.info(f"Dati non trovati. Avvio download da Kaggle: {COMPETITION_SLUG}...")
    
    # Autenticazione (legge os.environ automaticamente)
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        logger.error("Autenticazione Kaggle fallita. Verifica KAGGLE_USERNAME e KAGGLE_KEY nel .env")
        raise e

    os.makedirs(task_dir, exist_ok=True)
    
    try:
        # Scarica nella cartella target
        api.competition_download_files(COMPETITION_SLUG, path=task_dir, quiet=False)
        
        # Scompatta
        logger.info("Estrazione file...")
        zip_path = os.path.join(task_dir, f"{COMPETITION_SLUG}.zip")
        
        # A volte il file scaricato ha un nome diverso, cerchiamo lo zip
        if not os.path.exists(zip_path):
             files = [f for f in os.listdir(task_dir) if f.endswith('.zip')]
             if files:
                 zip_path = os.path.join(task_dir, files[0])

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(task_dir)
            
        logger.info("Download e estrazione completati.")
        
        # Pulizia opzionale dello zip
        os.remove(zip_path)
        
        return task_dir
        
    except Exception as e:
        logger.error(f"Errore durante il download da Kaggle: {e}")
        logger.info("SUGGERIMENTO: Hai accettato le regole della competizione sul sito Kaggle?")
        raise e

# -----------------------------------------------------------------------------
# Training Routine (DANN)
# -----------------------------------------------------------------------------
def train_one_epoch(model_wrapper, dataloader, optimizer, scheduler, device, 
                   epoch_idx, total_epochs, accumulation_steps=1):
    model_wrapper.train()
    running_loss = 0.0
    predictions, references = [], []
    
    optimizer.zero_grad()
    
    len_dataloader = len(dataloader)
    progress_bar = tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True)
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        lang_ids = batch["lang_ids"].to(device, non_blocking=True)
        
        # Calcolo dinamico Alpha (Annealing)
        current_step = step + epoch_idx * len_dataloader
        total_steps = total_epochs * len_dataloader
        p = float(current_step) / total_steps
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        with autocast(device_type=device.type, dtype=torch.float16):
            logits, loss = model_wrapper.forward(
                input_ids, attention_mask, labels=labels, 
                lang_ids=lang_ids, alpha=alpha
            )
            loss = loss / accumulation_steps

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), max_norm=1.0)
            optimizer.step()
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
        
        del input_ids, attention_mask, labels, lang_ids, logits, loss

    metrics = model_wrapper.compute_metrics(predictions, references)
    metrics["loss"] = running_loss / len(dataloader)
    return metrics, predictions, references

def manage_best_model(best_val_metric, val_metrics, model_wrapper, evaluation_metric, lower_is_better, best_model_state):
    current_metric = val_metrics[evaluation_metric]
    is_best = (current_metric < best_val_metric if lower_is_better else current_metric > best_val_metric)

    if is_best:
        logger.info(f"New Best Model! {evaluation_metric}: {current_metric:.4f} (Previous: {best_val_metric:.4f})")
        return current_metric, deepcopy(model_wrapper.state_dict())
    return best_val_metric, best_model_state

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    ConsoleUX.print_banner("SemEval 2026 Task 13 - Subtask A")

    # 1. Setup Data from Kaggle
    DATA_ROOT = os.getenv("DATA_PATH", "./data")
    # Questa funzione scarica i dati se non ci sono
    task_data_dir = setup_kaggle_data(DATA_ROOT) 

    # 2. Definisci i path corretti
    # Nota: Kaggle potrebbe scaricare file con nomi leggermente diversi o in sottocartelle
    # Controlla la struttura dello zip se ottieni FileNotFoundError
    train_path = os.path.join(task_data_dir, "train.parquet")
    val_path = os.path.join(task_data_dir, "validation.parquet")

    # Fallback se validation non esiste nel download (spesso Kaggle dà solo train/test)
    if not os.path.exists(val_path):
        logger.warning("Validation.parquet non trovato. Usando train.parquet come fallback o split manuale richiesto.")
        # Se necessario, qui potresti implementare uno split manuale del train
    
    with open("src_TaskA/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Override paths nel config
    config["data"]["train_path"] = train_path
    config["data"]["val_path"]   = val_path

    # 3. Experiment Setup
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME"),
        workspace=os.getenv("COMET_WORKSPACE"),
        auto_metric_logging=False
    )
    experiment.set_name(os.getenv("COMET_EXPERIMENT_NAME", "SemEval_Task13_DANN"))
    experiment.log_parameters(config)

    # 4. Device & Model
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Compute Device: {device}")
    
    writer = SummaryWriter(log_dir=config["training"].get("log_dir", "./results/logs"))

    logger.info(f"Initializing Model: {config['model']['model_name']}...")
    model_wrapper = CodeClassifier(config)
    model_wrapper.to(device)

    logger.info("Loading Datasets...")
    # Qui il load_data userà i path aggiornati da Kaggle
    train_dataset, val_dataset, _, _ = load_data(config, model_wrapper.tokenizer)

    num_workers = 2
    persistent_workers = True if num_workers > 0 else False
    
    train_dl = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], 
        shuffle=True, num_workers=num_workers, 
        pin_memory=False, persistent_workers=persistent_workers
    )
    val_dl = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], 
        shuffle=False, num_workers=num_workers, 
        pin_memory=False, persistent_workers=persistent_workers
    )

    # 5. Optimization
    optimizer = torch.optim.AdamW(
        model_wrapper.parameters(), 
        lr=float(config["training"]["learning_rate"]),
        weight_decay=0.01
    )
    
    accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
    num_epochs = config["training"].get("num_epochs", 5)
    total_steps = len(train_dl) * num_epochs // accumulation_steps
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(config["training"]["learning_rate"]),
        total_steps=total_steps, pct_start=config["training"].get("warmup_ratio", 0.1)
    )

    # 6. Training Loop
    lower_is_better = config["training"].get("best_metric_lower_is_better", False)
    best_val_metric = float("inf") if lower_is_better else float("-inf")
    best_model_state = None
    early_stop_patience = config["training"].get("early_stop_patience", 5)
    epochs_no_improve = 0

    ConsoleUX.print_banner(f"Starting Training for {num_epochs} Epochs")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_metrics, train_preds, train_refs = train_one_epoch(
            model_wrapper, train_dl, optimizer, scheduler, device, 
            epoch, num_epochs, accumulation_steps
        )

        val_metrics, val_preds, val_refs = evaluate(model_wrapper, val_dl, device)

        # Logging
        current_lr = scheduler.get_last_lr()[0]
        experiment.log_metric("Learning_Rate", current_lr, step=epoch)
        ConsoleUX.log_metrics("Train", train_metrics)
        ConsoleUX.log_metrics("Valid", val_metrics)

        for k, v in train_metrics.items():
            writer.add_scalar(f"Train/{k}", v, epoch)
            experiment.log_metric(f"Train/{k}", v, step=epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"Val/{k}", v, epoch)
            experiment.log_metric(f"Val/{k}", v, step=epoch)

        cm = confusion_matrix(val_refs, val_preds)
        experiment.log_confusion_matrix(matrix=cm, title=f"Confusion Matrix Epoch {epoch}")

        # Checkpointing
        eval_metric = config["training"]["evaluation_metric"]
        prev_best = best_val_metric
        best_val_metric, best_model_state = manage_best_model(
            best_val_metric, val_metrics, model_wrapper,
            eval_metric, lower_is_better, best_model_state
        )

        if best_val_metric == prev_best:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve}/{early_stop_patience} epochs.")
        else:
            epochs_no_improve = 0
            
        if epochs_no_improve >= early_stop_patience:
            ConsoleUX.print_banner(f"Early stopping triggered at Epoch {epoch+1}")
            break
            
        if device.type == 'mps':
            torch.mps.empty_cache()

    ConsoleUX.print_banner("Final Evaluation")
    
    if best_model_state is not None:
        logger.info("Loading best model weights...")
        model_wrapper.load_state_dict(best_model_state)
    
    test_metrics, preds, refs = evaluate(model_wrapper, val_dl, device)
    ConsoleUX.log_metrics("Final Test", test_metrics)
    print("\n" + classification_report(refs, preds))

    save_path = os.path.join(config["training"]["checkpoint_dir"], "best_model.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model_wrapper.state_dict(), save_path)
    logger.info(f"Best model saved to: {save_path}")

    writer.close()
    experiment.end()
    logger.info("Experiment completed successfully.")