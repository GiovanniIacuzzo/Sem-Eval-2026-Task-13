import os
import sys
import yaml
import json
import torch
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

from src.src_TaskB.models.model import CodeClassifier
from src.src_TaskB.utils.utils import set_seed
from src.src_TaskB.dataset.Inference_dataset import InferenceDataset

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

def inference_collate(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    extra_features = torch.stack([item['extra_features'] for item in batch])
    original_labels = [item['original_label'] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "extra_features": extra_features,
        "original_labels": original_labels
    }

# -----------------------------------------------------------------------------
# 2. MODEL LOADING
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
# 3. CASCADE INFERENCE LOGIC
# -----------------------------------------------------------------------------
def run_cascade_inference(binary_model, family_model, dataloader, device, family_id_to_name):
    final_predictions = []
    ground_truth = []
    
    logger.info("Running Cascade Inference...")
    pbar = tqdm(dataloader, desc="Inference", unit="batch")
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_features = batch["extra_features"].to(device)
            original_labels = batch["original_labels"]

            batch_size = input_ids.size(0)
            batch_preds = [""] * batch_size
            
            # --- STEP 1: BINARY ---
            out_bin = binary_model(input_ids, attention_mask, extra_features=extra_features)
            probs_bin = torch.softmax(out_bin[0], dim=1)
            preds_bin = torch.argmax(probs_bin, dim=1)
            
            human_indices = (preds_bin == 0).nonzero(as_tuple=True)[0]
            ai_indices = (preds_bin == 1).nonzero(as_tuple=True)[0]
            
            for idx in human_indices:
                batch_preds[idx.item()] = "Human"
                
            # --- STEP 2: FAMILIES ---
            if len(ai_indices) > 0:
                ai_input_ids = input_ids[ai_indices]
                ai_att_mask = attention_mask[ai_indices]
                ai_extra = extra_features[ai_indices]
                
                out_fam = family_model(ai_input_ids, ai_att_mask, extra_features=ai_extra)
                probs_fam = torch.softmax(out_fam[0], dim=1)
                preds_fam = torch.argmax(probs_fam, dim=1)
                
                for i, original_idx in enumerate(ai_indices):
                    fam_id = preds_fam[i].item()
                    fam_name = family_id_to_name.get(fam_id, f"Unknown_AI_{fam_id}")
                    batch_preds[original_idx.item()] = fam_name
            
            final_predictions.extend(batch_preds)
            ground_truth.extend(original_labels)

    return final_predictions, ground_truth

# -----------------------------------------------------------------------------
# 4. MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--ckpt_binary", type=str, required=True)
    parser.add_argument("--ckpt_families", type=str, required=True)
    parser.add_argument("--config", type=str, default="src/src_TaskB/config/config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/Task_B_Processed")
    parser.add_argument("--output_dir", type=str, default="results/TaskB_Cascade")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Config Loading
    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)
    config = raw_config["common"].copy()
    
    # Fix Params
    if "model_name" not in config:
        if "binary" in raw_config:
            config["model_name"] = raw_config["binary"]["model_name"]
            config["max_length"] = raw_config["binary"]["max_length"]
            config["batch_size"] = raw_config["binary"]["batch_size"]
        else:
            config["model_name"] = "microsoft/unixcoder-base"
            config["max_length"] = 512
            config["batch_size"] = 16

    # Mapping Loading
    mapping_path = os.path.join(args.data_dir, "family_mapping.json")
    with open(mapping_path, 'r') as f:
        fam_mapping = json.load(f)
    id_to_family = {v: k for k, v in fam_mapping.items()}

    # Init
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    logger.info("Loading Models...")
    model_binary = load_model_instance(args.ckpt_binary, config, "binary", device, num_labels=2)
    model_families = load_model_instance(args.ckpt_families, config, "families", device, len(id_to_family))

    logger.info(f"Loading Data: {args.test_file}")
    df_test = pd.read_parquet(args.test_file)
    
    dataset = InferenceDataset(df_test, tokenizer, max_length=config["max_length"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, collate_fn=inference_collate)

    # Run
    preds, truths = run_cascade_inference(model_binary, model_families, dataloader, device, id_to_family)

    # Export
    text_col = dataset.text_col # Recuperiamo quella usata dal dataset
    snippets = df_test[text_col].astype(str).str.slice(0, 50).tolist()
    
    results_df = pd.DataFrame({
        "text_snippet": snippets,
        "true_label": truths,
        "predicted_label": preds
    })
    results_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    # Metrics
    logger.info("-" * 50)
    logger.info("CLASSIFICATION REPORT")
    logger.info("-" * 50)
    
    truths = [str(t) for t in truths]
    preds = [str(p) for p in preds]
    
    # Stampa Report
    print(classification_report(truths, preds, digits=4, zero_division=0))
    
    # Matrice Confusione
    labels = sorted(list(set(truths + preds)))
    cm = confusion_matrix(truths, preds, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(args.output_dir, "confusion_matrix.csv"))

if __name__ == "__main__":
    main()