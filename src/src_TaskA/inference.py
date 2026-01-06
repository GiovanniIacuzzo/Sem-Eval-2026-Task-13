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
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn.utils.rnn import pad_sequence

sys.path.append(os.getcwd())

from src.src_TaskA.models.model import CodeModel
from src.src_TaskA.dataset.dataset import CodeDataset
from src.src_TaskA.utils.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DynamicCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        if 'labels' in batch[0] and batch[0]['labels'] is not None:
            labels = torch.stack([item['labels'] for item in batch])
        else:
            labels = None
            
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
    logger.info(f"🏗️  Building Model Architecture: {config['model_name']}...")
    
    model = CodeModel(config)
    
    logger.info(f"Loading LoRA adapters from {checkpoint_path}...")
    try:
        model.base_model.load_adapter(checkpoint_path, adapter_name="default")
    except Exception as e:
        logger.warning(f"Standard load_adapter failed ({e}). Trying PeftModel fallback...")
        model.base_model = PeftModel.from_pretrained(model.base_model.base_model, checkpoint_path)

    head_path = os.path.join(checkpoint_path, "classifier_head.pt")
    proj_path = os.path.join(checkpoint_path, "projector.pt")
    
    if os.path.exists(head_path):
        state_dict = torch.load(head_path, map_location=device, weights_only=True)
        model.classifier.load_state_dict(state_dict)
        logger.info("Classifier Head loaded.")
    else:
        logger.error(f"Classifier Head NOT FOUND at {head_path}!")
    
    if os.path.exists(proj_path):
        state_dict = torch.load(proj_path, map_location=device, weights_only=True)
        model.extra_projector.load_state_dict(state_dict)
        logger.info("Projector loaded.")
    else:
        logger.error(f"Projector NOT FOUND at {proj_path}!")
    
    model.to(device)
    model.eval()
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.to(dtype=dtype)
    
    return model

def run_inference(model, dataloader, device):
    all_preds = []
    all_probs = []
    all_labels = []
    has_labels = False
    
    logger.info("Starting Inference Loop...")
    
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(dataloader, desc="Predicting")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            extra_features = batch.get("extra_features", None)
            if extra_features is not None:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                extra_features = extra_features.to(device, dtype=dtype)
                
                # --- CHECK DIMENSIONALE DI SICUREZZA ---
                if i == 0:
                    feat_dim = extra_features.shape[1]
                    model_expect = model.extra_projector[0].in_features
                    if feat_dim != model_expect:
                        logger.error(f" FATAL MISMATCH: Dataset produces {feat_dim} features, Model expects {model_expect}!")
                        logger.error("Update dataset.py or model.py to match exactly.")
                        raise ValueError("Feature dimension mismatch")

            # Forward
            logits, _, _ = model(
                input_ids, attention_mask, extra_features=extra_features
            )
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.float().cpu().numpy())
            all_probs.extend(probs.float().cpu().numpy())
            
            if batch["labels"] is not None:
                has_labels = True
                all_labels.extend(batch["labels"].float().cpu().numpy())
            
    return all_preds, all_labels if has_labels else None, all_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/src_TaskA/config/config.yaml")
    parser.add_argument("--test_file", type=str, default="data/Task_A/test.parquet") 
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the checkpoint folder (e.g. results/.../best_model)")
    parser.add_argument("--output_file", type=str, default="predictions.csv")
    args = parser.parse_args()
    
    ConsoleUX.print_banner("Inference SemEval 2026")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)["common"]

    # 1. Load Data
    logger.info(f"Loading Test Data from {args.test_file}...")
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")
        
    df_test = pd.read_parquet(args.test_file)
    
    if 'code' not in df_test.columns and 'text' in df_test.columns:
        df_test = df_test.rename(columns={'text': 'code'})
    
    if 'label' not in df_test.columns:
        logger.info(" 'label' column missing. Running in BLIND TEST mode.")
        df_test['label'] = 0
        has_ground_truth = False
    else:
        df_test = df_test.dropna(subset=['label'])
        df_test['label'] = df_test['label'].astype(int)
        has_ground_truth = True
        
    logger.info(f"Test Samples: {len(df_test)}")

    # 2. Tokenizer & Dataset
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

    # 3. Model
    model = load_trained_model(config, args.checkpoint_dir, device)

    # 4. Inference
    preds, labels, probs = run_inference(model, test_dl, device)

    # 5. Report & Save
    logger.info("\n" + "="*50)
    
    # Se abbiamo le label, calcoliamo le metriche
    if has_ground_truth and labels is not None:
        logger.info("COMPUTING METRICS (Validation Mode)\n")
        label_names = ["Human", "AI"]
        try:
            report = classification_report(labels, preds, target_names=label_names, digits=4)
            print("\n" + report)
            cm = confusion_matrix(labels, preds)
            logger.info(f"Confusion Matrix:\n{cm}")
        except Exception as e:
            logger.warning(f"Could not compute metrics: {e}")
    else:
        logger.info("📦 GENERATING SUBMISSION (Blind Test Mode)")

    # 6. Save Predictions
    df_res = pd.DataFrame()
    if 'id' in df_test.columns:
        df_res['id'] = df_test['id']
    
    df_res["label"] = [int(p) for p in preds]
    df_res["prob_human"] = [p[0] for p in probs]
    df_res["prob_ai"] = [p[1] for p in probs]
    
    if has_ground_truth:
        df_res["true_label"] = labels

    df_res.to_csv(args.output_file, index=False)
    logger.info(f"Predictions saved to {args.output_file}")
    
    if 'id' in df_res.columns:
        sub_file = args.output_file.replace(".csv", "_submission.txt")
        df_res[['id', 'label']].to_csv(sub_file, index=False, sep='\t')
        logger.info(f"Submission file ready: {sub_file}")

class ConsoleUX:
    @staticmethod
    def print_banner(text):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

if __name__ == "__main__":
    main()