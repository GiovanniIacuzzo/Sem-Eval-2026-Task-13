import os
import logging
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from torch.amp import autocast
from dotenv import load_dotenv

# Local imports
from src_TaskB.models.model import CodeClassifier
# Importiamo FAMILY_MAP invece di GENERATOR_MAP
from src_TaskB.dataset.dataset import CodeDataset, load_base_dataframe, FAMILY_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

LABEL_NAMES = [
    "Human", "01-ai", "BigCode", "DeepSeek", "Gemma", "Phi", 
    "Llama", "Granite", "Mistral", "Qwen", "OpenAI"
]

# -----------------------------------------------------------------------------
# Inference Logic
# -----------------------------------------------------------------------------
def run_inference(model, test_df, config, device):
    model.eval()
    
    target_langs = config["model"].get("languages", [])
    language_map = {l: i for i, l in enumerate(target_langs)}
    
    dataset = CodeDataset(
        dataframe=test_df,
        tokenizer=model.tokenizer,
        language_map=language_map,
        max_length=config["data"]["max_length"],
        mode="val"
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config["training"].get("batch_size", 32), 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    all_preds, all_refs = [], []
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    dtype = torch.float16 if device_type == 'cuda' else torch.float32

    logger.info(f"Starting inference on {len(dataset)} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            lang_ids = batch.get("lang_ids", None)
            if lang_ids is not None:
                lang_ids = lang_ids.to(device)

            with autocast(device_type=device_type, dtype=dtype):
                # alpha=0.0 disabilita il ramo DANN
                logits, _ = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    lang_ids=lang_ids, 
                    alpha=0.0 
                )

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_refs.extend(labels.cpu().tolist())

    return all_preds, all_refs

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    
    config_path = "src_TaskB/config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Inizializzazione Modello (senza pesi per inferenza)
    model_wrapper = CodeClassifier(config)
    model_wrapper.to(device)

    # 2. Caricamento Pesi (Logica custom basata sul tuo save_checkpoint)
    checkpoint_dir = config["training"]["checkpoint_dir"]
    is_peft = config["model"].get("use_lora", True)

    if is_peft:
        logger.info(f"Loading LoRA adapters from {checkpoint_dir}")
        model_wrapper.base_model.load_adapter(checkpoint_dir, adapter_name="default")
        
        custom_path = os.path.join(checkpoint_dir, "custom_components.pt")
        if os.path.exists(custom_path):
            logger.info("Loading Custom Heads...")
            custom_state = torch.load(custom_path, map_location=device)
            model_wrapper.classifier.load_state_dict(custom_state['classifier'])
            model_wrapper.pooler.load_state_dict(custom_state['pooler'])
            model_wrapper.projection_head.load_state_dict(custom_state['projection_head'])
            model_wrapper.language_classifier.load_state_dict(custom_state['language_classifier'])
    else:
        full_path = os.path.join(checkpoint_dir, "full_model.bin")
        model_wrapper.load_state_dict(torch.load(full_path, map_location=device))

    # 3. Caricamento Dati
    test_path = "data/Task_B/test_sample.parquet" # Modifica con il path reale
    logger.info(f"Loading data from {test_path}...")
    test_df = load_base_dataframe(test_path)
    
    # Rimuovi eventuali campioni che non siamo riusciti a mappare (label -1)
    # a meno che non sia un test set "cieco" (dove label non esiste)
    if 'label' in test_df.columns and (test_df['label'] == -1).any():
        logger.warning("Filtering out samples with unknown family mapping.")
        test_df = test_df[test_df['label'] != -1].reset_index(drop=True)

    # 4. Esecuzione
    preds, refs = run_inference(model_wrapper, test_df, config, device)
    
    # 5. Report e Metriche
    print("\n" + "="*60)
    print("FINAL FAMILY CLASSIFICATION REPORT")
    print("="*60)
    
    # Filtriamo LABEL_NAMES se nel test set mancano classi (opzionale)
    present_labels = sorted(list(set(refs) | set(preds)))
    target_names = [LABEL_NAMES[i] for i in present_labels]

    print(classification_report(
        refs, 
        preds, 
        labels=present_labels,
        target_names=target_names,
        zero_division=0
    ))

    # 6. Salvataggio Risultati
    output_dir = "results_TaskB/inference_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(refs, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=target_names, yticklabels=target_names)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    # CSV di dettaglio
    results_df = pd.DataFrame({
        "True_Family": [LABEL_NAMES[r] if r != -1 else "Unknown" for r in refs],
        "Pred_Family": [LABEL_NAMES[p] for p in preds],
        "Correct": [r == p for r, p in zip(refs, preds)]
    })
    results_df.to_csv(os.path.join(output_dir, "family_predictions.csv"), index=False)
    
    logger.info(f"Inference complete. Results in {output_dir}")