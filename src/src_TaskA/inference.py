import os
import sys
import yaml
import torch
import logging
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.src_TaskA.models.model import HybridClassifier
from src.src_TaskA.dataset.dataset import AgnosticDataset
from src.src_TaskA.dataset.preprocess_features import AgnosticFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def prepare_test_data(test_path, config, device):
    """
    Controlla se il test set ha già le features. Se no, le estrae usando Qwen.
    """
    logger.info(f"Checking data: {test_path}")
    df = pd.read_parquet(test_path)
    
    if 'agnostic_features' in df.columns:
        logger.info("Features already present in dataset.")
        return df
    
    logger.info("Features missing. Initializing Feature Extractor (this takes time)...")
    
    cache_path = test_path.replace(".parquet", "_processed.parquet")
    if os.path.exists(cache_path):
        logger.info(f"Found cached processed file: {cache_path}")
        return pd.read_parquet(cache_path)

    extractor = AgnosticFeatureExtractor(config, str(device))
    
    features_list = []
    logger.info(f"Extracting features for {len(df)} test samples...")
    
    for code in tqdm(df['code'], desc="Feature Extraction"):
        try:
            feats = extractor.extract_all(code)
            features_list.append(feats)
        except Exception:
            features_list.append([0.0] * 11)
            
    df['agnostic_features'] = features_list
    
    logger.info(f"Saving processed test data to {cache_path}")
    df.to_parquet(cache_path)
    
    del extractor
    torch.cuda.empty_cache()
    
    return df

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference Device: {device}")

    # 1. Carica Configurazione
    config_path = os.path.join(args.checkpoint_dir, "config.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config not found in {args.checkpoint_dir}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Prepara i Dati
    test_df = prepare_test_data(args.test_file, config, device)
    
    if 'label' in test_df.columns:
        test_df = test_df.dropna(subset=['label']).reset_index(drop=True)
        has_labels = True
    else:
        has_labels = False
        test_df['label'] = 0 

    # 3. Carica Modello e Tokenizer
    logger.info("Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    
    model = HybridClassifier(config)
    
    weights_path = os.path.join(args.checkpoint_dir, "model_state.bin")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    # 4. DataLoader
    dataset = AgnosticDataset(test_df, tokenizer, max_length=config["data"]["max_length"], is_train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 5. Prediction Loop
    logger.info("Running Prediction...")
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            feats = batch["extra_features"].to(device)
            
            # Forward
            logits, _, _ = model(input_ids, mask, feats, labels=None)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            if has_labels:
                all_labels.extend(batch["labels"].numpy())

    # 6. Report Risultati
    if has_labels:
        print("\n" + "="*60)
        print("TEST SET EVALUATION REPORT".center(60))
        print("="*60)
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"\nAccuracy: {acc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=["Human", "AI"], digits=4))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        
        test_df['pred'] = all_preds
        errors = test_df[test_df['label'] != test_df['pred']]
        if not errors.empty:
            error_path = args.test_file.replace(".parquet", "_errors.csv")
            cols = ['code', 'label', 'pred']
            if 'language' in errors.columns: cols.append('language')
            errors[cols].head(100).to_csv(error_path, index=False)
            logger.info(f"Saved first {len(errors)} errors to {error_path}")

    else:
        print("\nInference Complete. Saving predictions...")
        test_df['prediction'] = all_preds
        test_df['probability_ai'] = [p[1] for p in all_probs]
        
        out_path = args.test_file.replace(".parquet", "_predictions.csv")
        test_df[['code', 'prediction', 'probability_ai']].to_csv(out_path, index=False)
        logger.info(f"Predictions saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="Path to test .parquet file")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the saved model folder")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    run_inference(args)