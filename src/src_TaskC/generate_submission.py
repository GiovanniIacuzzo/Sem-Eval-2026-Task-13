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

# -----------------------------------------------------------------------------
# 2. DATASET
# -----------------------------------------------------------------------------
class SubmissionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if "text" in df.columns: self.text_col = "text"
        elif "content" in df.columns: self.text_col = "content"
        elif "code" in df.columns: self.text_col = "code"
        else:
            self.text_col = df.select_dtypes(include=['object']).columns[0]
            logger.warning(f"Text column not found. Using '{self.text_col}'")

        if "id" in df.columns: self.id_col = "id"
        elif "ID" in df.columns: self.id_col = "ID"
        elif "index" in df.columns: self.id_col = "index"
        else:
            self.id_col = None
            logger.warning("ID column not found. Will use row index.")
            
    def _extract_stylistic_features(self, code):
        """
        Logica esatta del training Task C.
        """
        features = []
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        code_len = len(code) + 1
        
        # 1. Spazi
        features.append(code.count(' ') / code_len)
        # 2. Commenti
        features.append((code.count('#') + code.count('//')) / code_len)
        # 3. Punteggiatura
        features.append(len(re.findall(r'[{}()\[\];.,]', code)) / code_len)
        # 4. Lunghezza media riga
        avg_line_len = np.mean([len(l) for l in non_empty_lines]) if non_empty_lines else 0
        features.append(min(avg_line_len / 100.0, 1.0))
        # 5. Righe vuote
        features.append((len(lines) - len(non_empty_lines)) / (len(lines) + 1))
        # 6. Snake vs Camel
        snake_count = code.count('_')
        camel_count = len(re.findall(r'[a-z][A-Z]', code))
        features.append(snake_count / (snake_count + camel_count + 1))
        # 7. Keywords logiche
        logic_tokens = len(re.findall(r'\b(if|for|while|return|switch|case|break)\b', code))
        features.append(logic_tokens / (len(code.split()) + 1))
        # 8. Indentazione
        max_indent = 0
        if non_empty_lines:
            max_indent = max([len(l) - len(l.lstrip()) for l in non_empty_lines])
        features.append(min(max_indent / 20.0, 1.0))
        
        return torch.tensor(features, dtype=torch.float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = str(row[self.text_col])
        
        if self.id_col:
            sample_id = str(row[self.id_col])
        else:
            sample_id = str(idx)
        
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
            "id": sample_id
        }

def submission_collate(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    extra_features = torch.stack([item['extra_features'] for item in batch])
    ids = [item['id'] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "extra_features": extra_features,
        "ids": ids
    }

# -----------------------------------------------------------------------------
# 3. MODEL LOADER
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
# 4. PREDICTION LOGIC
# -----------------------------------------------------------------------------
def predict(model, dataloader, device):
    all_ids = []
    all_preds = []
    
    logger.info("Generating predictions...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Submission"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_features = batch["extra_features"].to(device)
            ids = batch["ids"]
            
            # Forward singolo
            logits, _, _ = model(
                input_ids, 
                attention_mask, 
                extra_features=extra_features
            )
            
            preds = torch.argmax(logits, dim=1)
            
            all_ids.extend(ids)
            all_preds.extend(preds.cpu().tolist())
            
    return all_ids, all_preds

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="Path to parquet test file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.bin")
    parser.add_argument("--config", type=str, default="src/src_TaskC/config/config.yaml")
    parser.add_argument("--output_file", type=str, default="submission_taskC.csv")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.config):
        logger.error(f"Config not found at {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Initializing Model...")
    model_name = config["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model(args.checkpoint, config, device)

    logger.info(f"Loading Test Data: {args.test_file}")
    df_test = pd.read_parquet(args.test_file)
    logger.info(f"Test samples: {len(df_test)}")
    
    dataset = SubmissionDataset(df_test, tokenizer, max_length=config["data"]["max_length"])
    dataloader = DataLoader(
        dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=4, 
        collate_fn=submission_collate
    )

    ids, labels = predict(model, dataloader, device)

    submission_df = pd.DataFrame({
        "ID": ids,
        "label": labels
    })
    
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    submission_df.to_csv(args.output_file, index=False)
    
    logger.info(f"Submission saved to {args.output_file}")
    
    print("\n--- PREVIEW ---")
    print(submission_df.head().to_string(index=False))
    
    counts = submission_df['label'].value_counts().sort_index()
    print("\n--- CLASS DISTRIBUTION ---")
    mapping = {0: "Human", 1: "Machine", 2: "Hybrid", 3: "Adversarial"}
    for label, count in counts.items():
        print(f"Class {label} ({mapping.get(label, '?')}): {count}")

if __name__ == "__main__":
    main()