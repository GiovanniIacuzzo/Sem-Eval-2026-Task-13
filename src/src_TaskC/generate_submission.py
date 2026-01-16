import os
import sys
import yaml
import torch
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.src_TaskC.models.model import CodeClassifier
from src.src_TaskC.utils.utils import set_seed
from src.src_TaskC.dataset.Inference_dataset import inference_collate,InferenceDataset

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
# 3. MODEL LOADER (FIXED KeyError 'model')
# -----------------------------------------------------------------------------
def load_model_instance(checkpoint_path, flat_config, num_labels, device):
    """
    Carica il modello costruendo la configurazione annidata corretta.
    """
    model_cfg = {
        "model": {
            "model_name": flat_config.get("model_name", "microsoft/unixcoder-base"),
            "num_labels": num_labels,
            "num_extra_features": flat_config.get("num_extra_features", 8),
            "gradient_checkpointing": False
        },
        "training": flat_config,
        "data": flat_config
    }
    
    logger.info(f"Loading Model from: {checkpoint_path} (Labels: {num_labels})")
    
    model = CodeClassifier(model_cfg)
    
    if os.path.isdir(checkpoint_path):
        weights_path = os.path.join(checkpoint_path, "best_model.bin")
        if not os.path.exists(weights_path):
             weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    else:
        weights_path = checkpoint_path
        
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)

    new_state_dict = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "") 
        if "loss_fn" in key: continue
        new_state_dict[key] = v
        
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        logger.warning(f"Strict loading failed ({e}), retrying non-strict...")
        model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 4. CASCADE PREDICTION
# -----------------------------------------------------------------------------
def predict_cascade(binary_model, attrib_model, dataloader, device):
    all_ids = []
    all_labels = []
    
    logger.info("Generating predictions (Human vs Machine -> AI/Hybrid/Adv)...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Submission"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_features = batch["extra_features"].to(device)
            ids = batch["ids"]
            
            batch_size = input_ids.size(0)
            
            # --- STAGE 1: Binary ---
            out_bin = binary_model(input_ids, attention_mask, extra_features=extra_features)
            preds_bin = torch.argmax(out_bin[0], dim=1)
            
            machine_indices = (preds_bin == 1).nonzero(as_tuple=True)[0]            
            batch_final_labels = [0] * batch_size 
            
            # --- STAGE 2: Attribution ---
            if len(machine_indices) > 0:
                mach_input_ids = input_ids[machine_indices]
                mach_att_mask = attention_mask[machine_indices]
                mach_extra = extra_features[machine_indices]
                
                out_attr = attrib_model(mach_input_ids, mach_att_mask, extra_features=mach_extra)
                preds_attr = torch.argmax(out_attr[0], dim=1)
                
                mapping = {0: 1, 1: 2, 2: 3}
                
                for i, idx_in_batch in enumerate(machine_indices):
                    local_label = preds_attr[i].item()
                    global_label = mapping.get(local_label, 1)
                    batch_final_labels[idx_in_batch.item()] = global_label
            
            all_ids.extend(ids)
            all_labels.extend(batch_final_labels)
            
    return all_ids, all_labels

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="Input parquet file")
    parser.add_argument("--ckpt_binary", type=str, required=True, help="Path to Stage 1 model")
    parser.add_argument("--ckpt_attrib", type=str, required=True, help="Path to Stage 2 model")
    parser.add_argument("--config", type=str, default="src/src_TaskC/config/config.yaml")
    parser.add_argument("--output_file", type=str, default="submission_taskC.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # 1. Config
    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)
    config = raw_config["common"].copy()
    
    config["model_name"] = raw_config.get("model", {}).get("model_name", "microsoft/unixcoder-base")
    config["max_length"] = raw_config.get("data", {}).get("max_length", 512)
    config["num_extra_features"] = raw_config.get("model", {}).get("num_extra_features", 8)
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # 2. Models
    # Stage 1: 2 Labels (Human vs Machine)
    model_binary = load_model_instance(args.ckpt_binary, config, 2, device)
    
    # Stage 2: 3 Labels (AI vs Hybrid vs Adv)
    model_attrib = load_model_instance(args.ckpt_attrib, config, 3, device)

    # 3. Data
    logger.info(f"Loading Submission Data: {args.test_file}")
    df_test = pd.read_parquet(args.test_file)
    
    dataset = InferenceDataset(df_test, tokenizer, max_length=config["max_length"])
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=inference_collate
    )

    # 4. Predict
    ids, labels = predict_cascade(model_binary, model_attrib, dataloader, device)

    # 5. Save
    logger.info(f"Saving submission to {args.output_file}...")
    
    submission_df = pd.DataFrame({
        "ID": ids,
        "label": labels
    })
    
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    submission_df.to_csv(args.output_file, index=False)
    
    logger.info("Submission Generated Successfully.")
    print("\nPreview:")
    print(submission_df.head().to_string(index=False))

if __name__ == "__main__":
    main()