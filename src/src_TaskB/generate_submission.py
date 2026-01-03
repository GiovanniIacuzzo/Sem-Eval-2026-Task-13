import os
import logging
import torch
import pandas as pd
import math
import collections
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None

from src.src_TaskB.models.model import CodeClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# PATHS & CONFIG
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

BINARY_CKPT = os.path.join(PROJECT_ROOT, "results/results_TaskB/checkpoints/binary")
FAMILIES_CKPT = os.path.join(PROJECT_ROOT, "results/results_TaskB/checkpoints/families")
TEST_PATH = os.path.join(PROJECT_ROOT, "data/Task_B/test.parquet")
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "results/results_TaskB/submission")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission_task_b.csv")

BINARY_CFG = {
    "model": {
        "model_name": "microsoft/unixcoder-base", 
        "num_labels": 2, 
        "use_lora": False,
        "num_extra_features": 5 
    }
}
FAMILIES_CFG = {
    "model": {
        "model_name": "microsoft/unixcoder-base", 
        "num_labels": 10, 
        "use_lora": True, 
        "lora_r": 64,
        "num_extra_features": 5 
    }
}

# -----------------------------------------------------------------------------
# STYLOMETRY & DATASET
# -----------------------------------------------------------------------------
def calculate_entropy(text):
    if not text: return 0.0
    counter = collections.Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def extract_stylometric_features(code):
    code_len = len(code)
    if code_len == 0: return [0.0] * 5
    lines = code.split('\n')
    num_lines = len(lines)
    avg_line_len = code_len / max(1, num_lines)
    special_chars = sum(1 for c in code if not c.isalnum() and not c.isspace())
    special_ratio = special_chars / code_len
    entropy = calculate_entropy(code)
    white_space_ratio = code.count(' ') / code_len
    return [math.log(code_len + 1), avg_line_len, special_ratio, entropy, white_space_ratio]

class SubmissionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info("Extracting stylometric features...")
        raw_features = [extract_stylometric_features(str(c)) for c in self.data['code']]
        scaler = StandardScaler()
        self.features = scaler.fit_transform(raw_features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = str(self.data.at[idx, 'code'])
        extra_feat = self.features[idx]

        encoding = self.tokenizer(
            code, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "extra_features": torch.tensor(extra_feat, dtype=torch.float)
        }

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
def load_model(checkpoint_dir, config, device, name):
    logger.info(f"[{name}] Loading model from {checkpoint_dir}...")
    if not os.path.exists(checkpoint_dir):
        if not os.path.exists(os.path.join(checkpoint_dir, "..")):
             raise FileNotFoundError(f"Checkpoint path invalid: {checkpoint_dir}")

    model = CodeClassifier(config)
    model.to(device)
    
    is_peft = config["model"]["use_lora"]
    if is_peft:
        try:
            model.base_model.load_adapter(checkpoint_dir, adapter_name="default")
            logger.info(f"[{name}] Adapter loaded.")
        except Exception as e:
            logger.warning(f"[{name}] Adapter load warning (might be embedded): {e}")
        
        cls_path = os.path.join(checkpoint_dir, "classifier.pt")
        custom_path = os.path.join(checkpoint_dir, "custom_components.pt")
        
        if os.path.exists(cls_path):
            state = torch.load(cls_path, map_location=device)
            model.classifier.load_state_dict(state)
            logger.info(f"[{name}] Head loaded from classifier.pt")
        elif os.path.exists(custom_path):
            state = torch.load(custom_path, map_location=device)
            if 'classifier' in state: model.classifier.load_state_dict(state['classifier'])
            if 'pooler' in state: model.pooler.load_state_dict(state['pooler'])
            logger.info(f"[{name}] Head loaded from custom_components.pt")
        else:
            logger.warning(f"[{name}] CRITICAL: No head weights found! Random init used.")
    else:
        full_path = os.path.join(checkpoint_dir, "full_model.bin")
        if not os.path.exists(full_path):
            full_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        
        if os.path.exists(full_path):
            model.load_state_dict(torch.load(full_path, map_location=device))
            logger.info(f"[{name}] Full weights loaded.")
        else:
            raise FileNotFoundError(f"Weights missing in {checkpoint_dir}")

    model.eval()
    return model

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Generating Submission using device: {device}")
    
    if not os.path.exists(TEST_PATH):
        TEST_PATH_SAMPLE = os.path.join(PROJECT_ROOT, "data/Task_B/test_sample.parquet")
        if os.path.exists(TEST_PATH_SAMPLE):
            logger.warning(f"Test file 'test.parquet' missing. Using 'test_sample.parquet' for demo.")
            TEST_PATH_ACTUAL = TEST_PATH_SAMPLE
        else:
            logger.error(f"No test file found at {TEST_PATH}")
            return
    else:
        TEST_PATH_ACTUAL = TEST_PATH

    df = pd.read_parquet(TEST_PATH_ACTUAL)
    df = df.reset_index(drop=True)
    logger.info(f"Loaded {len(df)} test samples.")
    
    # Identify ID Column
    possible_ids = ['id', 'ID', 'sample_id']
    id_col = next((col for col in possible_ids if col in df.columns), None)
    if not id_col:
        logger.warning("ID column missing. Generating sequential IDs.")
        df['id'] = df.index
        id_col = 'id'

    # --- FASE 1: Binary Classification ---
    logger.info(">>> STEP 1: Binary Classification (Human vs AI)")
    tok_bin = AutoTokenizer.from_pretrained(BINARY_CFG["model"]["model_name"])
    ds_bin = SubmissionDataset(df, tok_bin)
    dl_bin = DataLoader(ds_bin, batch_size=32, shuffle=False, num_workers=4)
    
    model_bin = load_model(BINARY_CKPT, BINARY_CFG, device, "Binary")
    
    is_ai_probs = []
    with torch.no_grad():
        for batch in tqdm(dl_bin, desc="Binary Infer"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            extra = batch["extra_features"].to(device)
            
            logits, _ = model_bin(input_ids, mask, extra_features=extra)
            probs = torch.softmax(logits, dim=1)
            is_ai_probs.extend(probs[:, 1].cpu().tolist())
    
    del model_bin
    torch.cuda.empty_cache()

    is_ai_preds = [1 if p > 0.5 else 0 for p in is_ai_probs]
    num_ai = sum(is_ai_preds)
    logger.info(f"Split Result -> Human: {len(df) - num_ai} | AI: {num_ai}")

    logger.info(">>> STEP 2: Families Classification (AI Attribution)")
    final_preds_map = {} 
    
    ai_indices = [i for i, x in enumerate(is_ai_preds) if x == 1]
    
    if len(ai_indices) > 0:
        df_ai = df.iloc[ai_indices].reset_index(drop=True)
        original_idx_map = df.iloc[ai_indices].index.tolist()
        
        tok_fam = AutoTokenizer.from_pretrained(FAMILIES_CFG["model"]["model_name"])
        ds_fam = SubmissionDataset(df_ai, tok_fam, max_length=512)
        dl_fam = DataLoader(ds_fam, batch_size=32, shuffle=False, num_workers=4)
        
        model_fam = load_model(FAMILIES_CKPT, FAMILIES_CFG, device, "Families")
        
        local_ptr = 0
        with torch.no_grad():
            for batch in tqdm(dl_fam, desc="Families Infer"):
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                extra = batch["extra_features"].to(device)
                
                logits, _ = model_fam(input_ids, mask, extra_features=extra)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                
                for p in preds:
                    orig_idx = original_idx_map[local_ptr]
                    final_preds_map[orig_idx] = p + 1
                    local_ptr += 1
        
        del model_fam
        torch.cuda.empty_cache()

    logger.info(">>> STEP 3: Saving Submission")
    final_labels = []
    
    for i in range(len(df)):
        if is_ai_preds[i] == 0:
            final_labels.append(0)
        else:
            final_labels.append(final_preds_map.get(i, 1))

    submission = pd.DataFrame({
        "ID": df[id_col],
        "label": final_labels
    })
    
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(SUBMISSION_FILE, index=False)
    
    logger.info(f"Submission saved successfully to: {SUBMISSION_FILE}")
    print("\nFinal Label Distribution:")
    print(submission['label'].value_counts().sort_index())

if __name__ == "__main__":
    main()