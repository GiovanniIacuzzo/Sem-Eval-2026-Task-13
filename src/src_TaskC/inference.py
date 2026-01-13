import os
import sys
import yaml
import torch
import argparse
import logging
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

from src.src_TaskC.models.model import CodeClassifier
from src.src_TaskC.utils.utils import set_seed

# -----------------------------------------------------------------------------
# 1. SETUP & LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

LABEL_MAPPING = {
    0: "Human",
    1: "AI-Generated",
    2: "Hybrid",
    3: "Adversarial"
}

# -----------------------------------------------------------------------------
# 2. INFERENCE DATASET
# -----------------------------------------------------------------------------
class InferenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.codes = dataframe['code'].astype(str).tolist()
        
        if 'label' in dataframe.columns:
            self.labels = dataframe['label'].astype(int).tolist()
        else:
            self.labels = [-1] * len(self.codes)

    def _extract_stylistic_features(self, code):
        """
        Deve essere ESATTAMENTE la stessa logica usata in training.
        """
        features = []
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        code_len = len(code) + 1
        
        features.append(code.count(' ') / code_len)
        features.append((code.count('#') + code.count('//')) / code_len)
        features.append(len(re.findall(r'[{}()\[\];.,]', code)) / code_len)
        avg_line_len = np.mean([len(l) for l in non_empty_lines]) if non_empty_lines else 0
        features.append(min(avg_line_len / 100.0, 1.0))
        features.append((len(lines) - len(non_empty_lines)) / (len(lines) + 1))
        snake_count = code.count('_')
        camel_count = len(re.findall(r'[a-z][A-Z]', code))
        features.append(snake_count / (snake_count + camel_count + 1))
        logic_tokens = len(re.findall(r'\b(if|for|while|return|switch|case|break)\b', code))
        features.append(logic_tokens / (len(code.split()) + 1))
        max_indent = 0
        if non_empty_lines:
            max_indent = max([len(l) - len(l.lstrip()) for l in non_empty_lines])
        features.append(min(max_indent / 20.0, 1.0))
        
        return torch.tensor(features, dtype=torch.float)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        
        style_feats = self._extract_stylistic_features(code)
        
        tokens = self.tokenizer.tokenize(code)
        capacity = self.max_length - 2 
        
        if len(tokens) <= capacity:
            chunk_str = code
        else:
            half = capacity // 2
            kept_tokens = tokens[:half] + tokens[-half:]
            chunk_str = self.tokenizer.convert_tokens_to_string(kept_tokens)

        encoding = self.tokenizer(
            chunk_str,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "extra_features": style_feats,
            "original_label": self.labels[idx],
            "text_snippet": code[:50]
        }

def inference_collate(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    extra_features = torch.stack([item['extra_features'] for item in batch])
    original_labels = [item['original_label'] for item in batch]
    snippets = [item['text_snippet'] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "extra_features": extra_features,
        "original_labels": original_labels,
        "snippets": snippets
    }

# -----------------------------------------------------------------------------
# 3. MODEL LOADING
# -----------------------------------------------------------------------------
def load_model(checkpoint_path, config, device):
    model_config = {
        "model": {
            "model_name": config["model"]["model_name"],
            "num_labels": 4,
            "num_extra_features": 8,
        },
        "training": config.get("training", {}),
    }
    
    model = CodeClassifier(model_config)
    
    logger.info(f"Loading weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if os.path.isdir(checkpoint_path):
        weights_path = os.path.join(checkpoint_path, "best_model.bin")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    else:
        weights_path = checkpoint_path
        
    state_dict = torch.load(weights_path, map_location=device)
    
    new_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model

# -----------------------------------------------------------------------------
# 4. INFERENCE LOOP
# -----------------------------------------------------------------------------
def run_inference(model, dataloader, device):
    predictions = []
    probabilities = []
    ground_truth = []
    snippets = []
    
    logger.info("Running Inference...")
    pbar = tqdm(dataloader, desc="Predicting", unit="batch")
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_features = batch["extra_features"].to(device)
            
            logits, _, _ = model(
                input_ids, 
                attention_mask, 
                extra_features=extra_features
            )
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            ground_truth.extend(batch["original_labels"])
            snippets.extend(batch["snippets"])

    return predictions, probabilities, ground_truth, snippets

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="Path to parquet file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.bin or dir)")
    parser.add_argument("--config", type=str, default="src/src_TaskC/config/config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/TaskC_Inference")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = load_model(args.checkpoint, config, device)
    
    logger.info(f"Reading Test Data: {args.test_file}")
    df_test = pd.read_parquet(args.test_file)
    
    dataset = InferenceDataset(df_test, tokenizer, max_length=config["data"]["max_length"])
    dataloader = DataLoader(
        dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=4, 
        collate_fn=inference_collate
    )
    
    preds, probs, truths, snippets = run_inference(model, dataloader, device)
    
    pred_labels = [LABEL_MAPPING[p] for p in preds]
    
    results_df = pd.DataFrame({
        "snippet": snippets,
        "true_label_id": truths,
        "pred_label_id": preds,
        "pred_label": pred_labels,
        "confidence": [max(p) for p in probs]
    })
    
    if all(t != -1 for t in truths):
        true_labels_str = [LABEL_MAPPING.get(t, "Unknown") for t in truths]
        
        logger.info("-" * 50)
        logger.info("CLASSIFICATION REPORT")
        logger.info("-" * 50)
        
        print(classification_report(true_labels_str, pred_labels, digits=4, zero_division=0))
        
        labels_list = ["Human", "AI-Generated", "Hybrid", "Adversarial"]
        cm = confusion_matrix(true_labels_str, pred_labels, labels=labels_list)
        cm_df = pd.DataFrame(cm, index=labels_list, columns=labels_list)
        cm_df.to_csv(os.path.join(args.output_dir, "confusion_matrix.csv"))
        logger.info(f"Confusion Matrix saved to {args.output_dir}")

    output_path = os.path.join(args.output_dir, "predictions.csv")
    results_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()