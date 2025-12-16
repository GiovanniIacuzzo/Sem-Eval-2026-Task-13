import os
import sys
import logging
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast
from dotenv import load_dotenv
from typing import List, Optional

from src_TaskA.models.model import CodeClassifier
from src_TaskA.dataset.Inference_dataset import InferenceDataset

# -----------------------------------------------------------------------------
# Configuration & UX
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConsoleUX:
    """Helper class for consistent console feedback."""
    @staticmethod
    def print_banner(text: str):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

# -----------------------------------------------------------------------------
# Submission Logic
# -----------------------------------------------------------------------------
def run_inference_pipeline(
    model_wrapper: CodeClassifier, 
    test_df: pd.DataFrame, 
    id_col_name: str, 
    output_file: str, 
    device: torch.device,
    batch_size: int = 32
):
    """
    Executes the full inference pipeline: Dataset prep -> Inference -> CSV Generation.
    """
    model_wrapper.eval()
    
    # Initialize Dataset
    dataset = InferenceDataset(
        dataframe=test_df, 
        tokenizer=model_wrapper.tokenizer, 
        max_length=model_wrapper.max_length,
        id_col=id_col_name
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=False, 
        persistent_workers=True if os.cpu_count() > 2 else False
    )

    ids: List[str] = []
    predictions: List[int] = []

    logger.info(f"Starting inference on {len(dataset)} samples...")

    # Determine precision settings based on device capabilities
    device_type = device.type if device.type in ["cuda", "mps"] else "cpu"
    dtype = torch.float16 if device_type != "cpu" else torch.float32

    # Inference Loop
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Generating Predictions", dynamic_ncols=True)
        
        for batch in progress_bar:
            # Non-blocking transfer to overlap I/O and Compute
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            batch_ids = batch["id"]

            # Mixed Precision Context for throughput optimization
            with autocast(device_type=device_type, dtype=dtype):
                logits, _ = model_wrapper(input_ids, attention_mask)
            
            # Greedy decoding (Argmax) for classification
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            
            ids.extend(batch_ids)
            predictions.extend(preds)

    # Artifact Generation - CORRETTO QUI
    # Usiamo id_col_name (es. "ID") invece di "id" fisso
    submission_df = pd.DataFrame({
        id_col_name: ids,  
        "label": predictions
    })
    
    # Ensure output directory hierarchy exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    # Write to disk
    submission_df.to_csv(output_file, index=False)
    
    logger.info(f"Submission artifact saved: {output_file}")
    print(f"\nPreview:\n{submission_df.head().to_string(index=False)}")

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------
def main():
    load_dotenv()
    # Prevent HuggingFace tokenizers from spawning threads (deadlock prevention)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    ConsoleUX.print_banner("SemEval Task 13 - Submission Generator")

    # 1. Configuration Validation
    config_path = "src_TaskA/config/config.yaml"
    if not os.path.exists(config_path):
        logger.critical(f"Config file missing: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Hardware Acceleration Setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    logger.info(f"Compute Device: {device}")

    # 3. Model Initialization
    logger.info("Initializing Model Architecture...")
    try:
        model_wrapper = CodeClassifier(config)
        model_wrapper.to(device)
    except Exception as e:
        logger.critical(f"Model initialization failed: {e}")
        sys.exit(1)

    # 4. Weight Injection
    checkpoint_path = os.path.join(config["training"]["checkpoint_dir"], "best_model.pt")
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading weights from: {checkpoint_path}")
        try:
            # Map location ensures weights are loaded to the correct device immediately
            state_dict = torch.load(checkpoint_path, map_location=device)
            model_wrapper.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.critical(f"State dict mismatch. Ensure architecture matches checkpoint. Error: {e}")
            sys.exit(1)
    else:
        logger.critical(f"Checkpoint not found: {checkpoint_path}. Aborting.")
        sys.exit(1)

    # 5. Data Ingestion
    test_path = "data/Task_A/test.parquet"
    
    if not os.path.exists(test_path):
        logger.error(f"Test dataset not found at: {test_path}")
        logger.info("Action Required: Download 'test.parquet' from Kaggle and place it in ./data/")
        sys.exit(1)

    logger.info(f"Loading Test Data: {test_path}")
    try:
        df = pd.read_parquet(test_path)
    except Exception as e:
        logger.critical(f"Parquet read error: {e}")
        sys.exit(1)
    
    # 6. Schema Validation (ID Column)
    possible_id_cols = ["id", "ID", "sample_id"]
    id_col_name = next((col for col in possible_id_cols if col in df.columns), None)
            
    if not id_col_name:
        logger.critical(f"Missing ID column. Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    logger.info(f"Target ID Column: '{id_col_name}'")

    # 7. Pre-processing
    df['code'] = df['code'].str.slice(0, 4096)

    # 8. Execution
    output_file = "./results_TaskA/submission/submission_task_a.csv"
    
    try:
        run_inference_pipeline(
            model_wrapper=model_wrapper, 
            test_df=df, 
            id_col_name=id_col_name, 
            output_file=output_file, 
            device=device,
            batch_size=32 
        )
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        sys.exit(1)

    ConsoleUX.print_banner("Process Completed Successfully")

if __name__ == "__main__":
    main()