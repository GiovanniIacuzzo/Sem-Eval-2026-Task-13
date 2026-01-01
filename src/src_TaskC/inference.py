import os
import sys
import logging
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List
from copy import deepcopy
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from src.src_TaskC.models.model import CodeClassifier
from src.src_TaskC.dataset.Inference_dataset import SlidingWindowDataset

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s", 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Model Loader
# -----------------------------------------------------------------------------
def load_ensemble_models(config, checkpoint_dir, device, k_folds=5) -> List[CodeClassifier]:
    models = []
    logger.info(f"Loading {k_folds} folds from {checkpoint_dir}...")
    config_clean = deepcopy(config)
    config_clean["model"]["use_lora"] = False 
    
    for fold in range(k_folds):
        fold_dir = os.path.join(checkpoint_dir, f"fold_{fold}")
        if not os.path.exists(fold_dir): 
            logger.warning(f"Fold {fold} missing, skipping.")
            continue
        
        model = CodeClassifier(config_clean)
        if config["model"].get("use_lora", False) and PEFT_AVAILABLE:
            try:
                model.base_model = PeftModel.from_pretrained(model.base_model, fold_dir)
                head_path = os.path.join(fold_dir, "head.pt")
                if os.path.exists(head_path):
                    model.classifier.load_state_dict(torch.load(head_path, map_location=device))
            except Exception as e:
                logger.error(f"Error loading LoRA fold {fold}: {e}")
                continue
        else:
            model_path = os.path.join(fold_dir, "model.pt")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
            
        model.to(device)
        model.eval()
        models.append(model)
    
    logger.info(f"Loaded {len(models)} models.")
    return models

# -----------------------------------------------------------------------------
# Evaluation Logic
# -----------------------------------------------------------------------------
def run_diagnostic_inference(models, test_df, id_col, label_col, output_dir, device, max_length):
    tokenizer = models[0].tokenizer
    
    dataset = SlidingWindowDataset(
        test_df, tokenizer, max_length, id_col, label_col, stride=384
    )
    
    ids = []
    preds = []
    trues = []
    confs = []
    n_chunks_list = []
    
    logger.info(f"Evaluating {len(dataset)} samples with Sliding Window...")
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Eval"):
            sample = dataset[i]
            chunks = sample["chunks"]
            label_true = sample["label"]
            
            encodings = tokenizer(
                chunks, 
                truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            
            # --- ENSEMBLE VOTING ---
            file_probs_sum = None
            
            for model in models:
                logits, _ = model(input_ids, attention_mask, alpha=0.0)
                chunk_probs = torch.softmax(logits, dim=1) 
                
                # Media sui chunk per questo modello
                model_file_prob = torch.mean(chunk_probs, dim=0)
                
                if file_probs_sum is None:
                    file_probs_sum = model_file_prob
                else:
                    file_probs_sum += model_file_prob
            
            # Media finale su ensemble
            final_probs = file_probs_sum / len(models)
            best_class = torch.argmax(final_probs).item()
            confidence = final_probs[best_class].item()
            
            ids.append(sample["id"])
            preds.append(best_class)
            trues.append(label_true)
            confs.append(confidence)
            n_chunks_list.append(sample["num_chunks"])

    # --- METRICS & REPORTING ---
    os.makedirs(output_dir, exist_ok=True)
    
    report = classification_report(trues, preds, digits=4)
    logger.info(f"\nClassification Report:\n{report}")
    
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    logger.info(f"Global Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    
    with open(os.path.join(output_dir, "report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\nMacro F1: {f1:.4f}")

    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Sliding Window) - F1: {f1:.3f}')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    results_df = pd.DataFrame({
        "id": ids,
        "true_label": trues,
        "pred_label": preds,
        "confidence": confs,
        "num_chunks": n_chunks_list,
        "is_correct": [t == p for t, p in zip(trues, preds)]
    })
    
    results_df.to_csv(os.path.join(output_dir, "all_predictions.csv"), index=False)
    
    errors_df = results_df[results_df["is_correct"] == False]
    errors_df.to_csv(os.path.join(output_dir, "errors_only.csv"), index=False)
    
    logger.info(f"Saved diagnostics to {output_dir}")
    logger.info(f"Total Errors: {len(errors_df)}/{len(dataset)}")
    
    long_files = results_df[results_df["num_chunks"] > 2]
    if not long_files.empty:
        acc_long = long_files["is_correct"].mean()
        logger.info(f"Accuracy on LONG files (>2 chunks): {acc_long:.4f} (Count: {len(long_files)})")
    
    short_files = results_df[results_df["num_chunks"] <= 2]
    if not short_files.empty:
        acc_short = short_files["is_correct"].mean()
        logger.info(f"Accuracy on SHORT files (<=2 chunks): {acc_short:.4f}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    
    config_path = "src/src_TaskC/config/config.yaml"
    checkpoint_dir = "results/results_TaskC/checkpoints"
    output_dir = "results/results_TaskC/inference_output"
    
    with open(config_path, "r") as f: config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = load_ensemble_models(config, checkpoint_dir, device)
    
    test_path = "data/Task_C/test_sample.parquet"
    if not os.path.exists(test_path):
        logger.error(f"Test sample not found at {test_path}")
        sys.exit(1)
        
    df = pd.read_parquet(test_path)
    logger.info(f"Loaded {len(df)} samples from {test_path}")
    
    id_col = next((c for c in ["id", "ID", "sample_id"] if c in df.columns), "id")
    label_col = next((c for c in ["label", "true_label"] if c in df.columns), "label")
    
    if label_col not in df.columns:
        logger.error("ERRORE: Non trovo la colonna delle label nel test_sample!")
        sys.exit(1)

    run_diagnostic_inference(
        models, df, id_col, label_col,
        output_dir, device, 
        config["data"]["max_length"]
    )