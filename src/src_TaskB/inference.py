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
from sklearn.metrics import classification_report, confusion_matrix

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
# 2. DATA UTILS
# -----------------------------------------------------------------------------
class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Identifica la colonna del testo
        self.text_col = "text" if "text" in df.columns else "content"
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        
        # Tokenizzazione
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
            "original_label": row["label"] if "label" in row else None
        }

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
# 3. MODEL LOADING
# -----------------------------------------------------------------------------
def load_model_instance(checkpoint_path, config, mode, device, num_labels):
    """Carica una singola istanza di modello (Binary o Families)."""
    
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
    
    # Gestione caricamento pesi
    if os.path.isdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, "model_state.bin")
        if os.path.exists(model_path):
             state_dict = torch.load(model_path, map_location=device)
        else:
            logger.warning("Standard binary not found, trying safetensors or adapters logic if implemented.")
            state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=device)
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)

    # Clean state dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "") 
        new_state_dict[k] = v
        
    try:
        model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        logger.error(f"Error loading weights: {e}")

    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 4. CASCADE INFERENCE LOGIC
# -----------------------------------------------------------------------------
def run_cascade_inference(binary_model, family_model, dataloader, device, family_id_to_name):
    final_predictions = []
    ground_truth = []
    
    logger.info("Running Cascade Inference...")
    
    # Barra di progresso
    pbar = tqdm(dataloader, desc="Inference", unit="batch")
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_features = batch["extra_features"].to(device)
            original_labels = batch["original_labels"]

            batch_size = input_ids.size(0)
            batch_preds = [""] * batch_size
            
            # --- STEP 1: BINARY MODEL ---
            # Output atteso: 0 = Human, 1 = AI
            out_bin = binary_model(input_ids, attention_mask, extra_features=extra_features)
            probs_bin = torch.softmax(out_bin[0], dim=1) # Logits position 0
            preds_bin = torch.argmax(probs_bin, dim=1)   # [Batch]
            
            # Identifichiamo gli indici classificati come AI
            ai_indices = (preds_bin == 1).nonzero(as_tuple=True)[0]
            human_indices = (preds_bin == 0).nonzero(as_tuple=True)[0]
            
            # Assegniamo subito "Human" a chi è stato predetto 0
            for idx in human_indices:
                batch_preds[idx.item()] = "Human"
                
            # --- STEP 2: FAMILIES MODEL ---
            if len(ai_indices) > 0:
                # Selezioniamo solo i sample classificati come AI
                ai_input_ids = input_ids[ai_indices]
                ai_att_mask = attention_mask[ai_indices]
                ai_extra = extra_features[ai_indices]
                
                out_fam = family_model(ai_input_ids, ai_att_mask, extra_features=ai_extra)
                probs_fam = torch.softmax(out_fam[0], dim=1)
                preds_fam = torch.argmax(probs_fam, dim=1)
                
                # Mappiamo ID -> Nome Famiglia
                for i, original_idx in enumerate(ai_indices):
                    fam_id = preds_fam[i].item()
                    fam_name = family_id_to_name.get(fam_id, f"Unknown_AI_{fam_id}")
                    batch_preds[original_idx.item()] = fam_name
            
            final_predictions.extend(batch_preds)
            ground_truth.extend(original_labels)

    return final_predictions, ground_truth

# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="Path to test_sample.parquet")
    parser.add_argument("--ckpt_binary", type=str, required=True, help="Path to Binary Model checkpoint")
    parser.add_argument("--ckpt_families", type=str, required=True, help="Path to Families Model checkpoint")
    parser.add_argument("--config", type=str, default="src/src_TaskB/config/config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/Task_B_Processed", help="Dir containing family_mapping.json")
    parser.add_argument("--output_dir", type=str, default="results/TaskB_Cascade")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Config & Mappings
    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)
        config = raw_config["common"]
    
    # Carichiamo il mapping delle famiglie (ID -> Nome)
    mapping_path = os.path.join(args.data_dir, "family_mapping.json")
    if not os.path.exists(mapping_path):
        logger.error(f"Mapping file not found at {mapping_path}. Cannot decode family IDs.")
        sys.exit(1)
        
    with open(mapping_path, 'r') as f:
        fam_mapping = json.load(f)
    
    # Invertiamo il mapping: {0: "GPT4", 1: "Llama"}
    id_to_family = {v: k for k, v in fam_mapping.items()}
    num_families = len(id_to_family)
    logger.info(f"Loaded {num_families} AI families.")

    # 2. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # 3. Load Models
    logger.info("Loading Binary Model...")
    model_binary = load_model_instance(args.ckpt_binary, config, "binary", device, num_labels=2)
    
    logger.info("Loading Families Model...")
    model_families = load_model_instance(args.ckpt_families, config, "families", device, num_labels=num_families)

    # 4. Load Data
    logger.info(f"Loading Test Data: {args.test_file}")
    df_test = pd.read_parquet(args.test_file)

    dataset = InferenceDataset(df_test, tokenizer, max_length=config["max_length"])
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=4, 
        collate_fn=inference_collate
    )

    # 5. Run Inference
    preds, truths = run_cascade_inference(model_binary, model_families, dataloader, device, id_to_family)

    # 6. Save & Evaluate
    logger.info("Saving results...")
    
    results_df = pd.DataFrame({
        "text_snippet": [t[:50] for t in df_test["text"] if "text" in df_test], # opzionale
        "true_label": truths,
        "predicted_label": preds
    })
    
    res_path = os.path.join(args.output_dir, "predictions.csv")
    results_df.to_csv(res_path, index=False)
    logger.info(f"Predictions saved to {res_path}")

    # Calcolo Metriche
    logger.info("-" * 50)
    logger.info("CLASSIFICATION REPORT")
    logger.info("-" * 50)
    
    # Ottieni tutte le label univoche presenti
    unique_labels = sorted(list(set(truths + preds)))
    
    print(classification_report(truths, preds, digits=4, zero_division=0))
    
    # Matrice di Confusione (salvata su file per leggibilità)
    cm = confusion_matrix(truths, preds, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    cm_path = os.path.join(args.output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    logger.info(f"Confusion Matrix saved to {cm_path}")

if __name__ == "__main__":
    main()