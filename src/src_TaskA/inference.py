import os
import sys
import yaml
import torch
import argparse
import logging
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, f1_score

from src.src_TaskA.models.model import CodeClassifier
from src.src_TaskA.dataset.dataset import CodeDataset
from src.src_TaskA.utils.utils import set_seed 

# -----------------------------------------------------------------------------
# 1. Logging
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 2. COLLATOR & UTILS
# -----------------------------------------------------------------------------
class InferenceCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        
        # Gestione Extra Features
        extra_features = None
        if 'extra_features' in batch[0] and batch[0]['extra_features'] is not None:
            extra_features = torch.stack([item['extra_features'] for item in batch])
        
        padded_inputs = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True, 
            return_tensors="pt"
        )

        return {
            "input_ids": padded_inputs["input_ids"],
            "attention_mask": padded_inputs["attention_mask"],
            "labels": labels,
            "extra_features": extra_features
        }

def load_model(config_path, checkpoint_path, device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["common"]
    
    model = CodeClassifier(config)
    
    logger.info(f"Loading weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
            
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        logger.warning(f"Strict loading failed, retrying with strict=False...")
        model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()
    
    return model, config

def run_inference(model, dataloader, device):
    all_preds = []
    all_labels = []
    
    logger.info("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            extra_features = batch.get("extra_features", None)
            if extra_features is not None:
                extra_features = extra_features.to(device)
            
            outputs = model(input_ids, attention_mask, labels=labels, extra_features=extra_features)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels)

# -----------------------------------------------------------------------------
# 3. MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="src/src_TaskA/config/config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/inference_TaskA")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Model
    model, config = load_model(args.config, args.checkpoint, device)
    
    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
    except ImportError:
        pass
        
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # 2. Data
    logger.info(f"Loading Test Data: {args.test_file}")
    df_test = pd.read_parquet(args.test_file)
    
    test_ds = CodeDataset(
        df_test, 
        tokenizer, 
        max_length=config["max_length"], 
        is_train=False,
        aug_config=None 
    )
    
    collate_fn = InferenceCollate(tokenizer)
    
    test_dl = DataLoader(
        test_ds, 
        batch_size=config["batch_size"] * 2,
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn
    )

    # 3. Inference
    preds, labels = run_inference(model, test_dl, device)

    # 4. Results
    logger.info("-" * 40)
    logger.info("GLOBAL RESULTS")
    logger.info("-" * 40)
    print(classification_report(labels, preds, target_names=["Human", "AI"], digits=4))
    
    # 5. Language Stats
    if 'language' in df_test.columns:
        logger.info("-" * 40)
        logger.info("PER-LANGUAGE PERFORMANCE")
        logger.info("-" * 40)
        df_test['pred'] = preds
        
        stats = []
        for lang in df_test['language'].unique():
            subset = df_test[df_test['language'] == lang]
            if len(subset) > 0:
                f1 = f1_score(subset['label'], subset['pred'], average='macro', zero_division=0)
                stats.append({"Language": lang, "Count": len(subset), "F1 Macro": f1})
            
        results_df = pd.DataFrame(stats).sort_values(by="F1 Macro", ascending=False)
        print(results_df.to_string(index=False))

    # Save Errors
    errors_df = df_test[df_test['pred'] != df_test['label']]
    error_path = os.path.join(args.output_dir, "errors.csv")
    errors_df.to_csv(error_path, index=False)
    logger.info(f"Saved {len(errors_df)} errors to {error_path}")

if __name__ == "__main__":
    main()