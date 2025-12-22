import os
import sys
import logging
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from torch.amp import autocast
from dotenv import load_dotenv
from typing import List, Dict, Tuple

# Gestione import se lanciato come script o modulo
try:
    from src.src_TaskA.models.model import CodeClassifier
    from src.src_TaskA.dataset.dataset import CodeDataset, load_and_preprocess
    from src.src_TaskA.utils.utils import compute_metrics
except ImportError:
    # Fallback per esecuzione diretta
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src_TaskA.models.model import CodeClassifier
    from src_TaskA.dataset.dataset import CodeDataset, load_and_preprocess
    from src_TaskA.utils.utils import compute_metrics

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# -----------------------------------------------------------------------------
# Logger & Console UX Setup
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
    def print_banner(text: str):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(metrics: Dict[str, float]):
        log_str = "[Results] "
        for k, v in metrics.items():
            log_str += f"{k.capitalize()}: {v:.4f} | "
        logger.info(log_str.strip(" | "))

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
def load_model_for_inference(config_path: str, checkpoint_dir: str, device: torch.device):
    """
    Carica il modello gestendo sia LoRA che Full Fine-Tuning.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Initializing Model Architecture...")
    model = CodeClassifier(config)
    
    # Percorsi possibili
    lora_adapter = checkpoint_dir # Peft cerca adapter_config.json qui
    heads_path = os.path.join(checkpoint_dir, "heads.pt")
    
    new_model_path = os.path.join(checkpoint_dir, "best_model_taskA.pt")
    old_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    is_lora = os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))
    
    if is_lora and PEFT_AVAILABLE:
        logger.info(f"Detected LoRA checkpoint in {checkpoint_dir}")
        model.base_model = PeftModel.from_pretrained(model.base_model, lora_adapter)
        
        if os.path.exists(heads_path):
            logger.info(f"Loading custom heads from {heads_path}...")
            heads_state = torch.load(heads_path, map_location=device)
            
            if 'classifier' in heads_state:
                model.classifier.load_state_dict(heads_state['classifier'])
            if 'pooler' in heads_state:
                model.pooler.load_state_dict(heads_state['pooler'])
            # NOTA: Rimosso caricamento projection_head che non esiste più
        else:
            logger.warning(f"Heads file ({heads_path}) NOT FOUND! Using random weights for classifier.")
            
    elif os.path.exists(new_model_path):
        logger.info(f"Loading Full Fine-Tuned model from {new_model_path}")
        state_dict = torch.load(new_model_path, map_location=device)
        # strict=False gestisce eventuali discrepanze minori (es. DANN head weights)
        model.load_state_dict(state_dict, strict=False)
        
    elif os.path.exists(old_model_path):
        logger.warning(f"Found OLD model {old_model_path}. Loading as fallback.")
        state_dict = torch.load(old_model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
            
    else:
        raise FileNotFoundError(f"No valid checkpoint found in {checkpoint_dir}")

    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Inference Pipeline
# -----------------------------------------------------------------------------
def run_inference(
    model_wrapper: CodeClassifier, 
    test_df: pd.DataFrame, 
    device: torch.device, 
    batch_size: int = 32
) -> Tuple[List[int], List[int], Dict[str, float]]:
    
    # Dummy language map per inference (DANN è disattivo, quindi non importa l'ID)
    dummy_map = {l: i for i, l in enumerate(test_df['language'].unique())}

    dataset = CodeDataset(
        dataframe=test_df,
        tokenizer=model_wrapper.tokenizer,
        language_map=dummy_map, 
        max_length=model_wrapper.max_length,
        augment=False 
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    all_preds, all_refs = [], []
    logger.info(f"Starting inference on {len(dataset)} samples...")

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    dtype = torch.float16 if device_type == 'cuda' else torch.float32

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast(device_type=device_type, dtype=dtype):
                # Alpha=0.0 disabilita DANN durante l'inferenza
                logits, _ = model_wrapper(input_ids, attention_mask, alpha=0.0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_refs.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_preds, all_refs)
    return all_preds, all_refs, metrics

# -----------------------------------------------------------------------------
# Analysis Tools
# -----------------------------------------------------------------------------
def analyze_per_language(df, preds, refs):
    """Calcola metriche separate per ogni linguaggio"""
    df_res = df.copy()
    df_res['pred'] = preds
    df_res['true'] = refs
    
    print("\n" + "="*60)
    print("PER-LANGUAGE BREAKDOWN".center(60))
    print("="*60)
    
    langs = df_res['language'].unique()
    results = []
    
    for lang in sorted(langs):
        subset = df_res[df_res['language'] == lang]
        if len(subset) == 0: continue
        
        acc = accuracy_score(subset['true'], subset['pred'])
        f1 = f1_score(subset['true'], subset['pred'], average='macro')
        count = len(subset)
        
        print(f"{lang:<15} | Samples: {count:<5} | Acc: {acc:.4f} | F1: {f1:.4f}")
        results.append({'lang': lang, 'f1': f1, 'acc': acc, 'count': count})
        
    print("="*60 + "\n")
    return pd.DataFrame(results)

def save_artifacts(preds, refs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(refs, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    report_df = pd.DataFrame({'True': refs, 'Pred': preds})
    report_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ConsoleUX.print_banner("SemEval Task 13 - Robust Inference")

    CONFIG_PATH = "src/src_TaskA/config/config.yaml"
    CHECKPOINT_DIR = "results/results_TaskA/checkpoints" 
    OUTPUT_DIR = "results/results_TaskA/inference_output"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # 1. Load Model
    try:
        model = load_model_for_inference(CONFIG_PATH, CHECKPOINT_DIR, device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # 2. Load Data
    # Priorità: Test Set dedicato -> Validation Set -> Train Set (Debug)
    TEST_FILE = "data/Task_A/test_sample.parquet"
    if not os.path.exists(TEST_FILE):
        logger.info("Test file not found, switching to Validation set...")
        TEST_FILE = "data/Task_A/validation.parquet"
    
    if not os.path.exists(TEST_FILE):
        logger.error("No dataset found for inference.")
        sys.exit(1)

    logger.info(f"Loading data from: {TEST_FILE}")
    test_df = load_and_preprocess(TEST_FILE)
    
    # 3. Inference
    preds, refs, metrics = run_inference(model, test_df, device)
    
    # 4. Results
    ConsoleUX.log_metrics(metrics)
    
    # 5. Detailed Breakdown
    analyze_per_language(test_df, preds, refs)
    
    print("\nGlobal Classification Report:")
    print(classification_report(refs, preds, target_names=["Human", "AI"]))
    
    save_artifacts(preds, refs, OUTPUT_DIR)
    logger.info(f"Artifacts saved to {OUTPUT_DIR}")