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
from typing import List

from peft import PeftModel

from src.src_TaskA.models.model import CodeClassifier
from src.src_TaskA.dataset.Inference_dataset import InferenceDataset

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
# Robust Model Loading Logic (LoRA + Custom Heads)
# -----------------------------------------------------------------------------
def load_model_for_submission(config_path: str, checkpoint_dir: str, device: torch.device):
    """
    Carica il modello per la sottomissione gestendo LoRA e Custom Heads.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Initializing Model Architecture...")
    model = CodeClassifier(config)
    
    adapter_path = checkpoint_dir
    heads_path = os.path.join(checkpoint_dir, "heads.pt")
    
    new_model_path = os.path.join(checkpoint_dir, "best_model_taskA.pt")
    old_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    is_lora = os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))
    
    if is_lora:
        logger.info(f"Detected LoRA checkpoint in {checkpoint_dir}")
        model.base_model = PeftModel.from_pretrained(model.base_model, adapter_path)
        
        if os.path.exists(heads_path):
            logger.info(f"Loading custom heads from {heads_path}...")
            heads_state = torch.load(heads_path, map_location=device, weights_only=False)
            
            if 'classifier' in heads_state:
                model.classifier.load_state_dict(heads_state['classifier'])
            if 'pooler' in heads_state:
                model.pooler.load_state_dict(heads_state['pooler'])
            if 'projection' in heads_state:
                model.projection_head.load_state_dict(heads_state['projection'])
        else:
            logger.warning(f"Heads file not found! Classifier will be RANDOM.")
            
    elif os.path.exists(new_model_path):
        logger.info(f"Loading NEW Full Model from {new_model_path}")
        state_dict = torch.load(new_model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        
    elif os.path.exists(old_model_path):
        logger.warning(f"Loading OLD Full Model from {old_model_path}")
        state_dict = torch.load(old_model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
            
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Submission
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
        pin_memory=True
    )

    ids: List[str] = []
    predictions: List[int] = []

    logger.info(f"Starting inference on {len(dataset)} samples...")

    device_type = "cuda" if device.type == "cuda" else "cpu"
    dtype = torch.float16 if device_type == "cuda" else torch.float32

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Generating Predictions", dynamic_ncols=True)
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            batch_ids = batch["id"]

            with autocast(device_type=device_type, dtype=dtype):
                logits, _ = model_wrapper(input_ids, attention_mask, alpha=0.0)
            
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            
            ids.extend(batch_ids)
            predictions.extend(preds)

    submission_df = pd.DataFrame({
        id_col_name: ids,  
        "label": predictions
    })
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    submission_df.to_csv(output_file, index=False)
    
    logger.info(f"Submission artifact saved: {output_file}")
    print(f"\nPreview:\n{submission_df.head().to_string(index=False)}")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    ConsoleUX.print_banner("SemEval Task 13 - Submission Generator")

    config_path = "src/src_TaskA/config/config.yaml"
    checkpoint_dir = "results/results_TaskA/checkpoints"
    
    if not os.path.exists(config_path):
        logger.critical(f"Config file missing: {config_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    try:
        model_wrapper = load_model_for_submission(config_path, checkpoint_dir, device)
    except Exception as e:
        logger.critical(f"Model initialization failed: {e}")
        logger.info("Tip: Did you complete the training successfully?")
        sys.exit(1)

    test_path = "data/Task_A/test.parquet"
    
    if not os.path.exists(test_path):
        logger.error(f"Test dataset not found at: {test_path}")
        if os.path.exists("data/Task_A/validation.parquet"):
            logger.warning("Falling back to validation.parquet for testing code logic...")
            test_path = "data/Task_A/validation.parquet"
        else:
            sys.exit(1)

    logger.info(f"Loading Test Data: {test_path}")
    df = pd.read_parquet(test_path)
    
    possible_id_cols = ["id", "ID", "sample_id"]
    id_col_name = next((col for col in possible_id_cols if col in df.columns), None)
            
    if not id_col_name:
        logger.warning("Missing ID column. Creating fake index IDs.")
        df["id"] = range(len(df))
        id_col_name = "id"
    
    logger.info(f"Target ID Column: '{id_col_name}'")

    df['code'] = df['code'].str.slice(0, 4096)

    output_file = "./results/results_TaskA/submission/submission_task_a.csv"
    
    run_inference_pipeline(
        model_wrapper=model_wrapper, 
        test_df=df, 
        id_col_name=id_col_name, 
        output_file=output_file, 
        device=device,
        batch_size=32 
    )

    ConsoleUX.print_banner("Process Completed Successfully")

if __name__ == "__main__":
    main()