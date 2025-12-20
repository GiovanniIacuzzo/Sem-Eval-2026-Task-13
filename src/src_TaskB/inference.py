import os
import logging
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None

from src.src_TaskB.models.model import CodeClassifier
from src.src_TaskB.dataset.Inference_dataset import InferenceDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

LABEL_NAMES = [
    "Human", "01-ai", "BigCode", "DeepSeek", "Gemma", "Phi", 
    "Llama", "Granite", "Mistral", "Qwen", "OpenAI"
]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
BINARY_CKPT = os.path.join(PROJECT_ROOT, "results/results_TaskB/checkpoints/binary")
FAMILIES_CKPT = os.path.join(PROJECT_ROOT, "results/results_TaskB/checkpoints/families")
TEST_FILE = os.path.join(PROJECT_ROOT, "data/Task_B/test_sample.parquet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results/results_TaskB/inference_output")

BINARY_CFG = {
    "model": {"model_name": "microsoft/codebert-base", "num_labels": 2, "use_lora": False}
}
FAMILIES_CFG = {
    "model": {"model_name": "microsoft/unixcoder-base", "num_labels": 10, "use_lora": True, "lora_r": 64}
}

def load_trained_model(checkpoint_dir, config, device, mode_name="Model"):
    logger.info(f"[{mode_name}] Loading from {checkpoint_dir}...")
    
    if not os.path.exists(checkpoint_dir):
        logger.error(f"PATH ERROR: {checkpoint_dir} does not exist.")
        raise FileNotFoundError(f"Directory not found: {checkpoint_dir}")

    model = CodeClassifier(config)
    model.to(device)
    
    is_peft = config["model"]["use_lora"]
    
    if is_peft:
        try:
            model.base_model.load_adapter(checkpoint_dir, adapter_name="default")
            logger.info(f"[{mode_name}] LoRA Adapters loaded successfully.")
        except Exception as e:
             raise RuntimeError(f"[{mode_name}] Failed to load LoRA: {e}")

        custom_path = os.path.join(checkpoint_dir, "custom_components.pt")
        if os.path.exists(custom_path):
            state = torch.load(custom_path, map_location=device)
            model.classifier.load_state_dict(state['classifier'])
            model.pooler.load_state_dict(state['pooler'])
            logger.info(f"[{mode_name}] Custom components loaded.")
        else:
            logger.warning(f"[{mode_name}] custom_components.pt missing. Using random init for heads.")

    else:
        full_path = os.path.join(checkpoint_dir, "full_model.bin")
        if not os.path.exists(full_path):
             fallback = os.path.join(checkpoint_dir, "pytorch_model.bin")
             if os.path.exists(fallback):
                 full_path = fallback
             else:
                 raise FileNotFoundError(f"[{mode_name}] Weights file not found at {full_path}")

        model.load_state_dict(torch.load(full_path, map_location=device))
        logger.info(f"[{mode_name}] Full weights loaded.")

    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running Inference on {device}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(TEST_FILE):
        logger.error(f"Test file not found: {TEST_FILE}")
        return

    df = pd.read_parquet(TEST_FILE)
    logger.info(f"Loaded {len(df)} samples from {TEST_FILE}")

    logger.info(">>> STEP A: Binary Classification")
    tok_bin = AutoTokenizer.from_pretrained(BINARY_CFG["model"]["model_name"])
    ds_bin = InferenceDataset(df, tok_bin)
    dl_bin = DataLoader(ds_bin, batch_size=32, shuffle=False, num_workers=2)
    
    model_bin = load_trained_model(BINARY_CKPT, BINARY_CFG, device, "Binary")
    
    is_ai_probs = []
    with torch.no_grad():
        for batch in tqdm(dl_bin, desc="Binary Infer"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits, _ = model_bin(input_ids, mask)
            probs = torch.softmax(logits, dim=1)
            is_ai_probs.extend(probs[:, 1].cpu().tolist())
            
    del model_bin
    torch.cuda.empty_cache()

    is_ai_preds = [1 if p > 0.5 else 0 for p in is_ai_probs]
    
    logger.info(">>> STEP B: AI Family Attribution")
    ai_indices = [i for i, x in enumerate(is_ai_preds) if x == 1]
    final_preds_map = {} 
    
    if len(ai_indices) > 0:
        df_ai = df.iloc[ai_indices].reset_index(drop=True)
        original_indices = df.iloc[ai_indices].index.tolist()
        
        tok_fam = AutoTokenizer.from_pretrained(FAMILIES_CFG["model"]["model_name"])
        ds_fam = InferenceDataset(df_ai, tok_fam)
        dl_fam = DataLoader(ds_fam, batch_size=32, shuffle=False, num_workers=2)
        
        model_fam = load_trained_model(FAMILIES_CKPT, FAMILIES_CFG, device, "Families")
        
        local_ptr = 0
        with torch.no_grad():
            for batch in tqdm(dl_fam, desc="Families Infer"):
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits, _ = model_fam(input_ids, mask)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                
                for p in preds:
                    orig_idx = original_indices[local_ptr]
                    final_preds_map[orig_idx] = p + 1 
                    local_ptr += 1
        
        del model_fam
        torch.cuda.empty_cache()

    final_predictions = []
    ground_truth = df['label'].tolist() if 'label' in df.columns else []
    
    for i in range(len(df)):
        if is_ai_preds[i] == 0:
            final_predictions.append(0)
        else:
            final_predictions.append(final_preds_map.get(i, 1))

    if len(ground_truth) > 0:
        labels_present = sorted(list(set(ground_truth) | set(final_predictions)))
        target_names = [LABEL_NAMES[i] for i in labels_present]
        
        print("\n" + "="*60)
        print("FINAL CASCADE REPORT")
        print("="*60)
        print(classification_report(ground_truth, final_predictions, labels=labels_present, target_names=target_names, digits=4))
        
        cm = confusion_matrix(ground_truth, final_predictions, labels=labels_present)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title("Confusion Matrix (Cascade)")
        
        out_img = os.path.join(OUTPUT_DIR, "confusion_matrix_cascade.png")
        plt.savefig(out_img)
        logger.info(f"Confusion Matrix saved to {out_img}")

    out_csv = os.path.join(OUTPUT_DIR, "predictions.csv")
    df_out = df.copy()
    df_out['pred'] = final_predictions
    df_out.to_csv(out_csv, index=False)
    logger.info(f"Predictions saved to {out_csv}")

if __name__ == "__main__":
    main()