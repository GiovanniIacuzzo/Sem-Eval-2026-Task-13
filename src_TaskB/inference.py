import os
import sys
import logging
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from torch.amp import autocast
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any

# Local imports
from src_TaskB.models.model import CodeClassifier
from src_TaskB.dataset.dataset import CodeDataset, load_and_preprocess, GENERATOR_MAP
from src_TaskB.utils.utils import compute_metrics

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
    """Helper class for consistent and readable console output."""
    @staticmethod
    def print_banner(text: str):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

    @staticmethod
    def log_metrics(metrics: Dict[str, float]):
        """Formats evaluation metrics into a clean log string."""
        log_str = "[Results] "
        for k, v in metrics.items():
            log_str += f"{k.capitalize()}: {v:.4f} | "
        logger.info(log_str.strip(" | "))

# -----------------------------------------------------------------------------
# Inference Logic
# -----------------------------------------------------------------------------
def run_inference(
    model_wrapper: CodeClassifier, 
    test_df: pd.DataFrame, 
    device: torch.device, 
    batch_size: int = 32
) -> Tuple[List[int], List[int], Dict[str, float]]:
    """
    Executes the inference pipeline on the provided test dataset.
    """
    model_wrapper.eval()

    # NOTE: Augmentation must be False during inference
    dataset = CodeDataset(
        dataframe=test_df,
        tokenizer=model_wrapper.tokenizer,
        max_length=model_wrapper.max_length,
        augment=False 
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=False
    )

    all_preds = []
    all_refs = []

    logger.info(f"Starting inference on {len(dataset)} samples...")

    device_type = device.type if device.type in ['cuda', 'mps'] else 'cpu'
    dtype = torch.float16 if device_type != 'cpu' else torch.float32

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast(device_type=device_type, dtype=dtype):
                logits, _ = model_wrapper(input_ids, attention_mask)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.detach().cpu().tolist())
            all_refs.extend(labels.detach().cpu().tolist())

    metrics = compute_metrics(all_preds, all_refs)
    return all_preds, all_refs, metrics

# -----------------------------------------------------------------------------
# Visualization & Reporting
# -----------------------------------------------------------------------------
def plot_confusion_matrix(y_true: List[int], y_pred: List[int], num_labels: int, save_path: str):
    """
    Generates and saves a Confusion Matrix heatmap.
    Handles both binary and multiclass scenarios automatically.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Calculate confusion matrix
    # labels=range(num_labels) ensures the matrix is always NxN even if some classes are missing
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
    
    # Adjust figure size for multiclass (Task B needs more space)
    figsize = (10, 8) if num_labels <= 10 else (24, 20)
    plt.figure(figsize=figsize)
    
    # Create labels list
    if num_labels == 2:
        tick_labels = ["Human", "AI"]
    else:
        # Reconstruct labels from GENERATOR_MAP
        inv_map = {v: k for k, v in GENERATOR_MAP.items()}
        # Get names sorted by index 0..N
        tick_labels = [inv_map.get(i, str(i)) for i in range(num_labels)]

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=tick_labels, yticklabels=tick_labels)
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Analysis")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion Matrix saved to: {save_path}")

def generate_error_report(
    df: pd.DataFrame, 
    y_true: List[int], 
    y_pred: List[int], 
    save_path: str,
    num_labels: int
) -> pd.DataFrame:
    """
    Exports misclassified samples to CSV for qualitative analysis.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    errors = []
    
    # Create label map for readability
    if num_labels == 2:
        label_map = {0: "Human", 1: "AI"}
    else:
        label_map = {v: k for k, v in GENERATOR_MAP.items()}

    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            row = df.iloc[idx]
            errors.append({
                "original_index": idx,
                "language": row.get("language", "N/A"),
                "true_label": f"{t} ({label_map.get(t, 'Unknown')})",
                "predicted_label": f"{p} ({label_map.get(p, 'Unknown')})",
                "code_snippet": row["code"][:200] + "..." 
            })

    error_df = pd.DataFrame(errors)
    error_df.to_csv(save_path, index=False)
    
    logger.info(f"Error Report generated: {save_path}")
    logger.info(f"Total Misclassifications: {len(error_df)} / {len(df)}")
    return error_df

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ConsoleUX.print_banner("SemEval Task 13 - Inference & Analysis")

    # 1. Configuration
    try:
        with open("src/config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Config file not found.")
        sys.exit(1)

    # 2. Device Setup
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Compute Device: {device}")

    # 3. Model Initialization
    logger.info("Initializing Model Architecture...")
    model_wrapper = CodeClassifier(config)
    model_wrapper.to(device)

    # 4. Weight Loading
    # NOTE: Ensure this points to the Task B checkpoint if you are running Task B inference!
    # For Task B you might want to hardcode or change this path temporarily
    checkpoint_path = os.path.join("results_task_b/checkpoints", "best_model.pt")
    
    if not os.path.exists(checkpoint_path):
        # Fallback to standard path if Task B specific path doesn't exist
        checkpoint_path = os.path.join(config["training"]["checkpoint_dir"], "best_model.pt")

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        sys.exit(1)
        
    logger.info(f"Loading weights from: {checkpoint_path}")
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model_wrapper.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        sys.exit(1)

    # 5. Data Loading
    DATA_PATH = os.getenv("DATA_PATH", "./data")
    # Usa sample per test veloce, o validation set per metriche complete
    TEST_FILE = "task_b/test_sample.parquet"  # Assumendo struttura data/task_b
    test_path = os.path.join(DATA_PATH, TEST_FILE)
    
    # Fallback search if file structure is different
    if not os.path.exists(test_path):
        TEST_FILE = "Task_B/test_sample.parquet"
        test_path = os.path.join(DATA_PATH, TEST_FILE)

    if not os.path.exists(test_path):
        logger.error(f"Test file not found at: {test_path}")
        sys.exit(1)

    # Determine Task Type for preprocessing based on config
    num_labels = config['model'].get('num_labels', 2)
    task_type = "multiclass" if num_labels > 2 else "binary"
    
    logger.info(f"Loading Test Dataset: {test_path} (Task: {task_type}, Labels: {num_labels})")
    
    # Use load_and_preprocess from dataset.py to handle labels correctly
    test_df = load_and_preprocess(test_path, task_type=task_type)
    
    # 6. Execution
    preds, refs, metrics = run_inference(model_wrapper, test_df, device, batch_size=32)

    # 7. Reporting
    ConsoleUX.print_banner("Final Results")
    ConsoleUX.log_metrics(metrics)

    # Generate labels list for report
    if num_labels == 2:
        target_names = ["Human", "AI"]
        all_labels = [0, 1]
    else:
        # Sort map by index to ensure names match classes
        sorted_gens = sorted(GENERATOR_MAP.items(), key=lambda x: x[1])
        target_names = [name for name, idx in sorted_gens]
        
        # Genera la lista completa degli indici [0, 1, ... 30]
        all_labels = list(range(num_labels))

        # Safety check if model has fewer labels than map
        if len(target_names) != num_labels:
             target_names = None 

    # FIX: Pass 'labels=all_labels' to force sklearn to report on all classes 
    # even if they are missing from the test sample.
    # 'zero_division=0' suppresses warnings for missing classes.
    print("\n" + classification_report(
        refs, 
        preds, 
        labels=all_labels, 
        target_names=target_names, 
        zero_division=0
    ))

    # Artifact Generation
    OUTPUT_DIR = "results/inference_analysis"
    
    plot_confusion_matrix(
        refs, preds, num_labels=num_labels,
        save_path=f"{OUTPUT_DIR}/confusion_matrix.png"
    )
    
    generate_error_report(
        test_df, refs, preds, 
        save_path=f"{OUTPUT_DIR}/error_analysis.csv",
        num_labels=num_labels
    )

    logger.info("Analysis completed successfully.")