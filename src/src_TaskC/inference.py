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
from sklearn.metrics import classification_report, confusion_matrix

from src.src_TaskC.models.model import CodeClassifier
from src.src_TaskC.utils.utils import set_seed
from src.src_TaskC.dataset.Inference_dataset import InferenceDataset, inference_collate

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

LABEL_NAMES = {
    0: "Human",
    1: "AI-Generated",
    2: "Hybrid",
    3: "Adversarial"
}

# -----------------------------------------------------------------------------
# 2. MODEL LOADING
# -----------------------------------------------------------------------------
def load_model_instance(checkpoint_path, config, num_labels, device):
    """Carica un modello specifico pulendo le chiavi non necessarie."""
    model_cfg = config.copy()
    model_cfg["model"]["num_labels"] = num_labels
    
    logger.info(f"Loading Model from: {checkpoint_path} (Labels: {num_labels})")
    
    model = CodeClassifier(model_cfg)
    
    if os.path.isdir(checkpoint_path):
        weights_path = os.path.join(checkpoint_path, "best_model.bin")
        if not os.path.exists(weights_path):
             weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    else:
        weights_path = checkpoint_path
        
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")
    
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "")
        
        if "loss_fn" in key:
            continue
            
        new_state_dict[key] = v
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        logger.warning(f"Strict loading failed slightly, verifying mismatch: {e}")
        model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 3. CASCADE INFERENCE
# -----------------------------------------------------------------------------
def run_inference(model_binary, model_attrib, dataloader, device):
    final_preds = []
    ground_truth = []
    
    logger.info("Running Hierarchical Inference (Binary -> Attribution)...")
    
    count_human = 0
    count_machine = 0
    
    for batch in tqdm(dataloader, desc="Inference", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        extra_features = batch["extra_features"].to(device)
        original_labels = batch["original_labels"]
        
        batch_size = input_ids.size(0)
        batch_results = [-1] * batch_size
        
        # --- STAGE 1: HUMAN vs MACHINE ---
        with torch.no_grad():
            logits_bin, _, _ = model_binary(input_ids, attention_mask, extra_features=extra_features)
            preds_bin = torch.argmax(logits_bin, dim=1)
        
        human_indices = (preds_bin == 0).nonzero(as_tuple=True)[0]
        machine_indices = (preds_bin == 1).nonzero(as_tuple=True)[0]
        
        count_human += len(human_indices)
        count_machine += len(machine_indices)
        
        for idx in human_indices:
            batch_results[idx.item()] = 0
            
        # --- STAGE 2: ATTRIBUTION ---
        if len(machine_indices) > 0:
            mach_input_ids = input_ids[machine_indices]
            mach_att_mask = attention_mask[machine_indices]
            mach_extra = extra_features[machine_indices]
            
            with torch.no_grad():
                logits_attr, _, _ = model_attrib(mach_input_ids, mach_att_mask, extra_features=mach_extra)
                preds_attr = torch.argmax(logits_attr, dim=1)
            
            mapping_local_to_global = {0: 1, 1: 2, 2: 3}
            
            for i, original_idx in enumerate(machine_indices):
                local_pred = preds_attr[i].item()
                global_pred = mapping_local_to_global.get(local_pred, 1)
                batch_results[original_idx.item()] = global_pred
        
        final_preds.extend(batch_results)
        ground_truth.extend(original_labels)
        
    logger.info(f"Stage 1 Stats: Predicted Human: {count_human} | Predicted Machine: {count_machine}")
    return final_preds, ground_truth

# -----------------------------------------------------------------------------
# 4. MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="Path al file parquet di test")
    parser.add_argument("--ckpt_binary", type=str, required=True, help="Path alla cartella/file del modello Binario")
    parser.add_argument("--ckpt_attrib", type=str, required=True, help="Path alla cartella/file del modello Attribution")
    parser.add_argument("--config", type=str, default="src/src_TaskC/config/config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/TaskC_Cascade")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # 1. Carica Configurazione
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    config["training"]["batch_size"] = args.batch_size
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    
    # 2. Carica Modelli
    # Modello 1: Binary (2 classi: Human vs Machine)
    model_binary = load_model_instance(args.ckpt_binary, config, num_labels=2, device=device)
    
    # Modello 2: Attribution (3 classi: AI, Hybrid, Adv)
    model_attrib = load_model_instance(args.ckpt_attrib, config, num_labels=3, device=device)
    
    # 3. Carica Dati
    logger.info(f"Loading Test Data: {args.test_file}")
    df_test = pd.read_parquet(args.test_file)
    
    dataset = InferenceDataset(df_test, tokenizer, max_length=config["data"]["max_length"])
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=inference_collate
    )
    
    # 4. Esegui Inferenza a Cascata
    preds, truths = run_inference(model_binary, model_attrib, dataloader, device)
    
    # 5. Export Risultati
    df_export = df_test.copy()
    if 'extra_features' in df_export.columns: 
        df_export = df_export.drop(columns=['extra_features'])
        
    df_export['predicted_label'] = preds
    df_export['predicted_name'] = df_export['predicted_label'].map(LABEL_NAMES)
    
    if 'label' in df_export.columns:
        df_export['true_name'] = df_export['label'].map(LABEL_NAMES)
        
        # Classification Report
        logger.info("\n" + classification_report(truths, preds, target_names=[LABEL_NAMES.get(i, str(i)) for i in range(4)], digits=4))
        
        # Matrice di Confusione
        cm = confusion_matrix(truths, preds)
        unique_labels = sorted(list(set(truths + preds)))
        labels_str = [LABEL_NAMES.get(i, str(i)) for i in unique_labels]
        
        cm_df = pd.DataFrame(cm, index=labels_str, columns=labels_str)
        cm_df.to_csv(os.path.join(args.output_dir, "confusion_matrix.csv"))
        logger.info(f"Confusion Matrix saved.")
    
    out_file = os.path.join(args.output_dir, "predictions_task_c.csv")
    df_export.to_csv(out_file, index=False)
    logger.info(f"Predictions saved to: {out_file}")

if __name__ == "__main__":
    main()