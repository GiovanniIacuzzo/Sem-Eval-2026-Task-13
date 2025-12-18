import os
import logging
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast
from dotenv import load_dotenv

from src_TaskB.models.model import CodeClassifier
from src_TaskB.dataset.Inference_dataset import InferenceDataset

# -----------------------------------------------------------------------------
# 2. Pipeline di Sottomissione
# -----------------------------------------------------------------------------
def run_submission_pipeline(model, test_df, config, device, id_col_name):
    model.eval()
    
    dataset = InferenceDataset(
        dataframe=test_df, 
        tokenizer=model.tokenizer, 
        max_length=config["data"]["max_length"],
        id_col=id_col_name
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config["training"].get("batch_size", 32), 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    all_ids = []
    all_preds = []

    logger.info(f"Inizio inferenza su {len(dataset)} campioni...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                logits, _ = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    alpha=0.0
                )
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            
            batch_ids = batch["id"]
            if torch.is_tensor(batch_ids):
                batch_ids = batch_ids.cpu().tolist()
            else:
                batch_ids = list(batch_ids)
            
            all_ids.extend(batch_ids)
            all_preds.extend(preds)

    return pd.DataFrame({"ID": all_ids, "label": all_preds})

# -----------------------------------------------------------------------------
# 3. Main
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    load_dotenv()
    
    config_path = "src_TaskB/config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_wrapper = CodeClassifier(config)
    model_wrapper.to(device)

    checkpoint_dir = config["training"]["checkpoint_dir"]
    is_peft = config["model"].get("use_lora", True)

    if is_peft:
        model_wrapper.base_model.load_adapter(checkpoint_dir, adapter_name="default")
        custom_path = os.path.join(checkpoint_dir, "custom_components.pt")
        if os.path.exists(custom_path):
            custom_state = torch.load(custom_path, map_location=device)
            model_wrapper.classifier.load_state_dict(custom_state['classifier'])
            model_wrapper.pooler.load_state_dict(custom_state['pooler'])
            model_wrapper.projection_head.load_state_dict(custom_state['projection_head'])
            model_wrapper.language_classifier.load_state_dict(custom_state['language_classifier'])
    else:
        full_path = os.path.join(checkpoint_dir, "full_model.bin")
        model_wrapper.load_state_dict(torch.load(full_path, map_location=device))

    test_path = "data/Task_B/test.parquet" 
    df_test = pd.read_parquet(test_path)
    
    id_col = next((c for c in ["ID", "id", "sample_id"] if c in df_test.columns), None)
    if id_col is None:
        df_test['ID'] = df_test.index
        id_col = 'ID'

    submission_df = run_submission_pipeline(model_wrapper, df_test, config, device, id_col)
    
    submission_df['ID'] = submission_df['ID'].astype(str).str.replace(r'tensor\(|\)', '', regex=True)
    
    output_path = "results_TaskB/submission/submission_task_b.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission pulita salvata in: {output_path}")