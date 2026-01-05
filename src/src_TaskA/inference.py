import os
import sys
import yaml
import torch
import argparse
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.getcwd())

from src.src_TaskA.models.model import CodeModel
from src.src_TaskA.dataset.dataset import CodeDataset
from src.src_TaskA.utils.utils import set_seed

from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class DynamicCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch])
        extra_features = torch.stack([item['extra_features'] for item in batch])
        
        padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        return {
            "input_ids": padded_ids,
            "attention_mask": padded_mask,
            "labels": labels,
            "extra_features": extra_features
        }

def load_trained_model(config, checkpoint_path, device):
    logger.info(f"Loading base model: {config['model_name']}...")
    model = CodeModel(config)
    
    logger.info(f"Loading LoRA adapters from {checkpoint_path}...")
    try:
        model.base_model.load_adapter(checkpoint_path, adapter_name="default")
    except Exception as e:
        logger.error(f"Fallback loading: {e}")
        model.base_model = PeftModel.from_pretrained(model.base_model.base_model, checkpoint_path)

    head_path = os.path.join(checkpoint_path, "classifier_head.pt")
    proj_path = os.path.join(checkpoint_path, "projector.pt")
    
    if os.path.exists(head_path):
        state_dict = torch.load(head_path, map_location=device, weights_only=True)
        model.classifier.load_state_dict(state_dict)
    
    if os.path.exists(proj_path):
        state_dict = torch.load(proj_path, map_location=device, weights_only=True)
        model.extra_projector.load_state_dict(state_dict)
    
    model.to(device, dtype=torch.bfloat16)
    model.eval()
    return model

def run_inference(model, dataloader, device):
    all_preds = []
    all_labels = []
    all_probs = []
    
    logger.info("Starting Inference...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            extra_features = batch.get("extra_features", None)

            if extra_features is not None:
                extra_features = extra_features.to(device, dtype=torch.bfloat16)
            
            logits, _, _ = model(
                input_ids, attention_mask, extra_features=extra_features
            )
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.float().cpu().numpy())
            all_labels.extend(labels.float().cpu().numpy())
            all_probs.extend(probs.float().cpu().numpy())
            
    return all_preds, all_labels, all_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/src_TaskA/config/config.yaml")
    parser.add_argument("--test_file", type=str, default="data/Task_A/test_sample.parquet")
    parser.add_argument("--checkpoint_dir", type=str, default="results/results_TaskA/checkpoints/best_model")
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)["common"]

    logger.info(f"Loading Test Data from {args.test_file}...")
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")
        
    df_test = pd.read_parquet(args.test_file)
    if 'code' not in df_test.columns and 'text' in df_test.columns:
        df_test = df_test.rename(columns={'text': 'code'})
    
    df_test = df_test.dropna(subset=['code', 'label'])
    df_test['label'] = df_test['label'].astype(int)
    logger.info(f"Test Samples: {len(df_test)}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    test_ds = CodeDataset(df_test, tokenizer, max_length=config["max_length"], is_train=False)
    collate_fn = DynamicCollate(tokenizer)
    
    test_dl = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    model = load_trained_model(config, args.checkpoint_dir, device)

    preds, labels, probs = run_inference(model, test_dl, device)

    logger.info("\n" + "="*50)
    logger.info("FINAL TEST RESULTS\n")
    logger.info("="*50)
    
    label_names = ["Human", "AI"]
    
    try:
        report = classification_report(labels, preds, target_names=label_names, digits=4)
        print("\n" + report)
        
        cm = confusion_matrix(labels, preds)
        logger.info(f"Confusion Matrix:\n{cm}")
    except Exception as e:
        logger.warning(f"Could not compute metrics: {e}")
    
    # Save Predictions
    output_path = "predictions_taskA.csv"
    df_res = pd.DataFrame({
        "True Label": labels,
        "Predicted Label": preds,
        "Prob_Human": [p[0] for p in probs],
        "Prob_AI": [p[1] for p in probs]
    })
    df_res.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()