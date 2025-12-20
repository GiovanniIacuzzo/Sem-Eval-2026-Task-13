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
from copy import deepcopy
from typing import List

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from src.src_TaskC.models.model import CodeClassifier
from src.src_TaskC.dataset.dataset import CodeDataset, load_and_preprocess
from src.src_TaskC.utils.utils import compute_metrics

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Ensemble Model Loader
# -----------------------------------------------------------------------------
def load_models_for_ensemble(config, checkpoint_dir, device, k_folds=5) -> List[CodeClassifier]:
    """
    Carica i 5 modelli correggendo il bug della doppia inizializzazione LoRA.
    """
    models = []
    
    for fold in range(k_folds):
        fold_dir = os.path.join(checkpoint_dir, f"fold_{fold}")
        if not os.path.exists(fold_dir):
            logger.warning(f"Checkpoint for Fold {fold} not found in {fold_dir}. Skipping.")
            continue
            
        logger.info(f"Loading Fold {fold} model from {fold_dir}...")
        
        config_clean = deepcopy(config)
        config_clean["model"]["use_lora"] = False 
        
        model = CodeClassifier(config_clean)
        
        if config["model"].get("use_lora", False) and PEFT_AVAILABLE:
            logger.info(f"Applying LoRA adapter for Fold {fold}...")
            model.base_model = PeftModel.from_pretrained(model.base_model, fold_dir)
            
            head_path = os.path.join(fold_dir, "head.pt")
            if os.path.exists(head_path):
                heads_state = torch.load(head_path, map_location=device)
                model.classifier.load_state_dict(heads_state)
                logger.info("Custom classifier head loaded.")
            else:
                logger.error(f"Head weights missing for Fold {fold}!")
        else:
            model_path = os.path.join(fold_dir, "model.pt")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            
        model.to(device)
        model.eval()
        models.append(model)
    
    if not models:
        raise ValueError("No models loaded! Check your checkpoint directory.")
        
    return models

# -----------------------------------------------------------------------------
# Ensemble Inference
# -----------------------------------------------------------------------------
def run_ensemble_inference(
    models: List[CodeClassifier], 
    test_df: pd.DataFrame, 
    device: torch.device, 
    batch_size: int = 32
):
    tokenizer = models[0].tokenizer
    dummy_lang_map = {l: i for i, l in enumerate(config["model"]["languages"])}
    
    dataset = CodeDataset(
        dataframe=test_df,
        tokenizer=tokenizer,
        language_map=dummy_lang_map, 
        max_length=config["data"]["max_length"],
        augment=False 
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    all_probs = []
    all_refs = []
    
    logger.info(f"Starting ENSEMBLE inference on {len(dataset)} samples with {len(models)} models...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            batch_probs_sum = None
            
            for model in models:
                logits, _ = model(input_ids, attention_mask, alpha=0.0)
                probs = torch.softmax(logits, dim=1)
                
                if batch_probs_sum is None:
                    batch_probs_sum = probs
                else:
                    batch_probs_sum += probs
            
            avg_probs = batch_probs_sum / len(models)
            
            all_probs.extend(avg_probs.cpu().tolist())
            all_refs.extend(labels.cpu().tolist())

    final_preds = np.argmax(all_probs, axis=1)
    
    metrics = compute_metrics(final_preds, all_refs)
    return final_preds, all_refs, metrics

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def save_artifacts(preds, refs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(refs, preds)
    target_names = ["Human", "AI", "Hybrid", "Adversarial"]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix (Ensemble)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    report_df = pd.DataFrame({'True': refs, 'Pred': preds})
    label_map = {0: "Human", 1: "AI", 2: "Hybrid", 3: "Adv"}
    report_df['True_Label'] = report_df['True'].map(label_map)
    report_df['Pred_Label'] = report_df['Pred'].map(label_map)
    
    report_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    logger.info(f"Artifacts saved in {output_dir}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    CONFIG_PATH = "src/src_TaskC/config/config.yaml"
    CHECKPOINT_DIR = "results/results_TaskC/checkpoints"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(CONFIG_PATH):
        logger.error("Config not found.")
        sys.exit(1)
        
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    try:
        models = load_models_for_ensemble(config, CHECKPOINT_DIR, device, k_folds=5)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        sys.exit(1)

    TEST_FILE = "data/Task_C/test_sample.parquet" 
    
    if not os.path.exists(TEST_FILE):
        logger.error(f"Test file not found: {TEST_FILE}")
        sys.exit(1)

    logger.info(f"Loading Test Data from: {TEST_FILE}")
    test_df = load_and_preprocess(TEST_FILE)
    
    preds, refs, metrics = run_ensemble_inference(models, test_df, device)
    
    print("\n" + "="*40)
    print("ENSEMBLE RESULTS (5 FOLDS)")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    print("\nClassification Report:")
    target_names = ["Human", "AI", "Hybrid", "Adv"]
    print(classification_report(refs, preds, target_names=target_names))
    
    save_artifacts(preds, refs, "results/results_TaskC/inference_output")