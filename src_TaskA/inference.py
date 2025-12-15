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
from typing import List, Dict, Tuple

from src_TaskA.models.model import CodeClassifier
from src_TaskA.dataset.dataset import CodeDataset, load_and_preprocess
from src_TaskA.utils.utils import compute_metrics

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
    
    Optimizations:
    - Disables Gradient Calculation: Reduces memory usage and computation time.
    - Mixed Precision (MPS/CUDA): Utilizes FP16 for faster tensor operations.
    - Increased Batch Size: Feasible since no gradients are stored.
    """
    model_wrapper.eval()

    # NOTE: Augmentation must be False during inference to ensure 
    # deterministic evaluation and fair comparison.
    dataset = CodeDataset(
        dataframe=test_df,
        tokenizer=model_wrapper.tokenizer,
        max_length=model_wrapper.max_length,
        augment=False 
    )
    
    # DataLoader Configuration:
    # - num_workers=2: Pre-fetches data on CPU while GPU processes current batch.
    # - pin_memory=False: Disabled for MPS compatibility (avoids warning).
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

    # Determine device type for autocast context
    device_type = device.type if device.type in ['cuda', 'mps'] else 'cpu'
    dtype = torch.float16 if device_type != 'cpu' else torch.float32

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing", dynamic_ncols=True):
            # Non-blocking transfer for asynchronous I/O
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Mixed Precision Context
            with autocast(device_type=device_type, dtype=dtype):
                logits, _ = model_wrapper(input_ids, attention_mask)

            # Get class predictions
            preds = torch.argmax(logits, dim=1)

            # Detach and move to CPU immediately to free VRAM
            all_preds.extend(preds.detach().cpu().tolist())
            all_refs.extend(labels.detach().cpu().tolist())

    metrics = compute_metrics(all_preds, all_refs)
    return all_preds, all_refs, metrics

# -----------------------------------------------------------------------------
# Visualization & Reporting
# -----------------------------------------------------------------------------
def plot_confusion_matrix(y_true: List[int], y_pred: List[int], labels: List[int], save_path: str):
    """
    Generates and saves a Confusion Matrix heatmap.
    Visualizes True Positives, False Positives, False Negatives, and True Negatives.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Analysis")
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion Matrix saved to: {save_path}")

def generate_error_report(
    df: pd.DataFrame, 
    y_true: List[int], 
    y_pred: List[int], 
    save_path: str
) -> pd.DataFrame:
    """
    Exports misclassified samples to CSV for qualitative analysis.
    Useful for identifying patterns in model errors (e.g., specific languages).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    errors = []
    
    label_map = {0: "Human", 1: "AI"}

    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            row = df.iloc[idx]
            errors.append({
                "original_index": idx,
                "language": row.get("language", "N/A"),
                "true_label": f"{t} ({label_map.get(t, '?')})",
                "predicted_label": f"{p} ({label_map.get(p, '?')})",
                "code_snippet": row["code"][:200] + "..." # Truncate for readability
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
    # Suppress HuggingFace parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ConsoleUX.print_banner("SemEval Task 13 - Inference & Analysis")

    # 1. Configuration
    try:
        with open("src_TaskA/config/config.yaml", "r") as f:
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
    checkpoint_path = os.path.join(config["training"]["checkpoint_dir"], "best_model.pt")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        sys.exit(1)
        
    logger.info(f"Loading weights from: {checkpoint_path}")
    try:
        # Load state dict strictly to ensure architecture match
        state_dict = torch.load(checkpoint_path, map_location=device)
        model_wrapper.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        sys.exit(1)

    # 5. Data Loading
    test_path = "data/Task_A/test_sample.parquet"

    if not os.path.exists(test_path):
        logger.error(f"Test file not found: {test_path}")
        sys.exit(1)

    logger.info(f"Loading Test Dataset: {test_path}")
    test_df = load_and_preprocess(test_path)
    
    # 6. Execution
    preds, refs, metrics = run_inference(model_wrapper, test_df, device, batch_size=32)

    # 7. Reporting
    ConsoleUX.print_banner("Final Results")
    ConsoleUX.log_metrics(metrics)

    print("\n" + classification_report(refs, preds, target_names=["Human (0)", "AI (1)"]))

    # Artifact Generation
    OUTPUT_DIR = "results_taskA/inference_analysis"
    
    plot_confusion_matrix(
        refs, preds, labels=[0, 1], 
        save_path=f"{OUTPUT_DIR}/confusion_matrix.png"
    )
    
    generate_error_report(
        test_df, refs, preds, 
        save_path=f"{OUTPUT_DIR}/error_analysis.csv"
    )

    logger.info("Analysis completed successfully.")