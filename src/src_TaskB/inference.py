import os
import sys
import yaml
import json
import torch
import gc
import argparse
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from src.src_TaskB.models.model import CodeClassifier
from src.src_TaskB.utils.utils import set_seed

# -----------------------------------------------------------------------------
# Configurazione Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Funzioni di Utilità per Label
# -----------------------------------------------------------------------------
def get_family_mapping(data_dir):
    """Carica il mapping generato durante il training per le famiglie."""
    mapping_path = os.path.join(data_dir, "family_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        # Invertiamo per avere {Stringa: ID}
        # Nota: nel training il mapping salvato è {Label: ID} solitamente
        return mapping
    else:
        logger.error(f"CRITICAL: Mapping file not found at {mapping_path}")
        sys.exit(1)

def prepare_ground_truth(df, family_mapping):
    """
    Crea due vettori di verità (Ground Truth) partendo dal dataframe.
    Assumiamo che la colonna 'label' contenga la stringa della famiglia (es: 'Human', 'GPT-4', etc.)
    """
    # 1. Ground Truth Binary
    # Human = 0, Qualsiasi AI = 1
    # Adatta la stringa 'Human' se nel tuo dataset è diversa (es. 'human', 'Human-Eval', ecc.)
    y_true_binary = df['label'].apply(lambda x: 0 if str(x).lower() == 'human' else 1).values
    
    # 2. Ground Truth Families
    # Usa il mapping per convertire la stringa in ID
    # Se una label nel test non esiste nel mapping (caso raro), mettiamo -1 o gestiamo l'errore
    y_true_families = df['label'].map(family_mapping).fillna(-1).astype(int).values
    
    return y_true_binary, y_true_families

# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
        
        if "extra_features" in row:
             item["extra_features"] = torch.tensor(row["extra_features"], dtype=torch.float)
        
        return item

# -----------------------------------------------------------------------------
# Caricamento Modello
# -----------------------------------------------------------------------------
def load_model_for_inference(config, mode, device):
    checkpoint_dir = os.path.join(config["training"]["checkpoint_dir"], mode)
    
    # Istanzia wrapper
    model = CodeClassifier(config, class_weights=None)
    
    if config["model"]["use_lora"]:
        logger.info(f"[{mode}] Loading LoRA Adapter...")
        from peft import PeftModel
        model.base_model = PeftModel.from_pretrained(model.base_model, checkpoint_dir)
        
        # Carica Classifier Head
        head_path = os.path.join(checkpoint_dir, "classifier.pt")
        if os.path.exists(head_path):
            model.classifier.load_state_dict(torch.load(head_path, map_location=device))
        else:
            logger.warning(f"[{mode}] Classifier head not found at {head_path}!")
    else:
        logger.info(f"[{mode}] Loading Full Model...")
        model_path = os.path.join(checkpoint_dir, "full_model.bin")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Motore di Inferenza
# -----------------------------------------------------------------------------
def run_evaluation(mode, df, y_true, args, raw_config, device):
    """
    Esegue l'inferenza per un task specifico (binary o families)
    """
    print(f"\n{'='*40}\nEVALUATING TASK: {mode.upper()}\n{'='*40}")
    
    # 1. Configurazione Specifica
    mode_config = raw_config["common"].copy()
    if mode in raw_config:
        mode_config.update(raw_config[mode])
        
    # Recupera le label names corrette per il report
    data_dir = mode_config.get("data_dir", "data/Task_B_Processed")
    
    if mode == "binary":
        label_names = ["Human", "AI"]
        num_labels = 2
    else:
        # Per families ricarichiamo il mapping per avere l'ordine corretto dei nomi
        fam_map = get_family_mapping(data_dir)
        # Ordina le label in base all'ID (value)
        label_names = [k for k, v in sorted(fam_map.items(), key=lambda item: item[1])]
        num_labels = len(label_names)

    final_config = {
        "model": {
            "model_name": mode_config["model_name"],
            "num_labels": num_labels,
            "num_extra_features": 8,
            "use_lora": mode_config.get("use_lora", False),
            "lora_r": mode_config.get("lora_r", 32),
            "lora_dropout": mode_config.get("lora_dropout", 0.1),
            "class_weights": False
        },
        "training": mode_config,
        "data": mode_config
    }

    # 2. Carica Modello e Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_config["model_name"])
    model = load_model_for_inference(final_config, mode, device)

    # 3. DataLoader
    dataset = InferenceDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=mode_config["batch_size"]*2, shuffle=False, num_workers=4)

    # 4. Predizione
    preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference {mode}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra = batch.get("extra_features", None)
            if extra is not None: extra = extra.to(device)

            logits, _, _ = model(input_ids, attention_mask, extra_features=extra)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)

    # 5. Calcolo Metriche
    preds = np.array(preds)
    
    # Filtra eventuali -1 nei target (se c'erano label sconosciute)
    valid_indices = y_true != -1
    y_true_clean = y_true[valid_indices]
    preds_clean = preds[valid_indices]

    acc = accuracy_score(y_true_clean, preds_clean)
    report = classification_report(y_true_clean, preds_clean, target_names=label_names)
    cm = confusion_matrix(y_true_clean, preds_clean)

    print(f"\nAccuracy ({mode}): {acc:.4f}")
    print(f"\nClassification Report ({mode}):\n")
    print(report)
    print(f"\nConfusion Matrix ({mode}):\n{cm}")

    # 6. Pulizia Memoria
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/src_TaskB/config/config.yaml")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test_sample.parquet")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Carica Configurazione
    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)

    # 1. Carica il DataFrame una volta per tutte
    logger.info(f"Loading Test Data: {args.test_file}")
    df_test = pd.read_parquet(args.test_file)
    logger.info(f"Loaded {len(df_test)} samples.")

    # 2. Prepara le Ground Truth per entrambi i task
    # Serve il mapping per il task families per convertire le stringhe in ID
    data_dir_families = raw_config["families"].get("data_dir", "data/Task_B_Processed")
    family_mapping = get_family_mapping(data_dir_families)
    
    logger.info("Generating Ground Truths for Binary and Families tasks...")
    y_true_binary, y_true_families = prepare_ground_truth(df_test, family_mapping)

    # 3. Esegui Task BINARY
    run_evaluation("binary", df_test, y_true_binary, args, raw_config, device)

    # 4. Esegui Task FAMILIES
    run_evaluation("families", df_test, y_true_families, args, raw_config, device)
    
    logger.info("Final Evaluation Completed.")