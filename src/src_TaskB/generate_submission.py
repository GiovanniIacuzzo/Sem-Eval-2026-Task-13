import os
import sys
import yaml
import json
import torch
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.src_TaskB.models.model import CodeClassifier
from src.src_TaskB.utils.utils import set_seed

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
# 2. KAGGLE MAPPING TRANSLATION
# -----------------------------------------------------------------------------
def convert_to_kaggle_id(internal_name):
    """
    Traduce il nome usato internamente dal nostro modello nell'ID richiesto da Kaggle.
    """
    name = str(internal_name).lower().strip()
    
    mapping = {
        "human": 0,
        "deepseek": 1,
        "qwen": 2,
        "yi": 3,
        "starcoder": 4,
        "gemma": 5,
        "phi": 6,
        "llama": 7,
        "granite": 8,
        "mistral": 9,
        "gpt": 10,
        "openai": 10,
        "other": 0
    }
    
    if name in mapping:
        return mapping[name]
    else:
        logger.warning(f"Warning: Unknown label '{internal_name}' mapped to 0 (Human)")
        return 0

# -----------------------------------------------------------------------------
# 3. DATASET
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
            raise ValueError(f"Columns error. Found: {list(df.columns)}")

        if "id" in df.columns: self.id_col = "id"
        elif "ID" in df.columns: self.id_col = "ID"
        elif "index" in df.columns: self.id_col = "index"
        else:
            raise ValueError(f"ID column missing. Found: {list(df.columns)}")
            
        logger.info(f"Text Column: '{self.text_col}' | ID Column: '{self.id_col}'")
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        sample_id = str(row[self.id_col]) 
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        extra_feats = torch.zeros(8)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "extra_features": extra_feats,
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
# 4. MODEL LOADER
# -----------------------------------------------------------------------------
def load_model_instance(checkpoint_path, config, mode, device, num_labels):
    model_config = {
        "model": {
            "model_name": config["model_name"],
            "num_labels": num_labels,
            "num_extra_features": config.get("num_extra_features", 8),
            "use_lora": config.get("use_lora", False),
        },
        "training": config,
        "data": config
    }
    
    model = CodeClassifier(model_config)
    
    logger.info(f"[{mode.upper()}] Loading weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if os.path.isdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, "model_state.bin")
        if os.path.exists(model_path):
             state_dict = torch.load(model_path, map_location=device)
        else:
            state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=device)
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "") 
        new_state_dict[k] = v
        
    try:
        model.load_state_dict(new_state_dict, strict=False)
    except Exception:
        pass 

    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 5. CASCADE PREDICTION LOGIC
# -----------------------------------------------------------------------------
def predict_cascade(binary_model, family_model, dataloader, device, id_to_family):
    all_ids = []
    all_kaggle_ids = []
    
    logger.info("Generating predictions...")
    
    

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Submission"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_features = batch["extra_features"].to(device)
            ids = batch["ids"]
            
            batch_size = input_ids.size(0)
            
            # 1. Binary Step
            out_bin = binary_model(input_ids, attention_mask, extra_features=extra_features)
            probs_bin = torch.softmax(out_bin[0], dim=1)
            preds_bin = torch.argmax(probs_bin, dim=1)
            
            ai_indices = (preds_bin == 1).nonzero(as_tuple=True)[0]
            
            # Default: tutti Human (ID 0)
            batch_kaggle_ids = [0] * batch_size 
            
            # 2. Family Step (Solo AI)
            if len(ai_indices) > 0:
                ai_input_ids = input_ids[ai_indices]
                ai_att_mask = attention_mask[ai_indices]
                ai_extra = extra_features[ai_indices]
                
                out_fam = family_model(ai_input_ids, ai_att_mask, extra_features=ai_extra)
                probs_fam = torch.softmax(out_fam[0], dim=1)
                preds_fam = torch.argmax(probs_fam, dim=1) # ID interno del modello
                
                for i, idx_in_batch in enumerate(ai_indices):
                    fam_id_internal = preds_fam[i].item()
                    
                    # A. Ottieni nome interno
                    fam_name_internal = id_to_family.get(fam_id_internal, "other")
                    
                    # B. Traduci in ID Kaggle
                    kaggle_id = convert_to_kaggle_id(fam_name_internal)
                    
                    batch_kaggle_ids[idx_in_batch.item()] = kaggle_id
            
            all_ids.extend(ids)
            all_kaggle_ids.extend(batch_kaggle_ids)
            
    return all_ids, all_kaggle_ids

# -----------------------------------------------------------------------------
# 6. MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="Path to test.parquet")
    parser.add_argument("--ckpt_binary", type=str, required=True)
    parser.add_argument("--ckpt_families", type=str, required=True)
    parser.add_argument("--config", type=str, default="src/src_TaskB/config/config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/Task_B_Processed")
    parser.add_argument("--output_file", type=str, default="submission.csv")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Config
    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)
    config = raw_config["common"].copy()
    
    if "model_name" not in config:
        if "binary" in raw_config:
            config["model_name"] = raw_config["binary"]["model_name"]
            config["max_length"] = raw_config["binary"]["max_length"]
            config["batch_size"] = raw_config["binary"]["batch_size"]
        else:
            config["model_name"] = "microsoft/unixcoder-base"
            config["max_length"] = 512
            config["batch_size"] = 16

    # 2. Mapping
    mapping_path = os.path.join(args.data_dir, "family_mapping.json")
    if not os.path.exists(mapping_path):
        logger.error(f"Mapping file not found at {mapping_path}")
        sys.exit(1)
        
    with open(mapping_path, 'r') as f:
        fam_mapping = json.load(f)
    
    id_to_family = {v: k for k, v in fam_mapping.items()}
    logger.info(f"Loaded {len(fam_mapping)} AI families mapping.")

    # 3. Models
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    logger.info("Loading Binary Model...")
    model_binary = load_model_instance(args.ckpt_binary, config, "binary", device, num_labels=2)
    
    logger.info("Loading Families Model...")
    model_families = load_model_instance(args.ckpt_families, config, "families", device, num_labels=len(fam_mapping))

    # 4. Data
    logger.info(f"Loading Submission Data: {args.test_file}")
    df_test = pd.read_parquet(args.test_file)
    
    dataset = SubmissionDataset(df_test, tokenizer, max_length=config["max_length"])
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=4, 
        collate_fn=submission_collate
    )

    # 5. Predict with Translation
    ids, kaggle_labels = predict_cascade(model_binary, model_families, dataloader, device, id_to_family)

    # 6. Save
    logger.info(f"Saving submission to {args.output_file}...")
    
    submission_df = pd.DataFrame({
        "ID": ids,
        "label": kaggle_labels
    })
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    submission_df.to_csv(args.output_file, index=False)
    
    logger.info("Done.")
    print("\nPreview (Kaggle IDs):")
    print(submission_df.head().to_string(index=False))

if __name__ == "__main__":
    main()