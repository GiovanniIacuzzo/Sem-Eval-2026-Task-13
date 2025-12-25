import os
import sys
import logging
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from dotenv import load_dotenv

try:
    from src.src_TaskA.models.model import FusionCodeClassifier
    from src.src_TaskA.features.stylometry import StylometryExtractor
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src_TaskA.models.model import FusionCodeClassifier
    try:
        from src_TaskA.features.stylometry import StylometryExtractor
    except ImportError:
        StylometryExtractor = None

# -----------------------------------------------------------------------------
# Configuration & UX
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. DATASET SPECIFICO PER SUBMISSION
# -----------------------------------------------------------------------------
class SubmissionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 512, id_col: str = "id"):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id_col = id_col
        
        self.lang_map = {'c#': 0, 'c++': 1, 'go': 2, 'java': 3, 'javascript': 4, 'php': 5, 'python': 6, 'unknown': 7}
        
        self.stylo_extractor = StylometryExtractor() if StylometryExtractor else None

    def __len__(self):
        return len(self.data)

    def _clean_code(self, code: str) -> str:
        if not isinstance(code, str): return ""
        return code.replace("```", "").strip()

    def __getitem__(self, idx):
        row_id = self.data.at[idx, self.id_col]
        
        code = self._clean_code(str(self.data.at[idx, "code"]))
        if len(code) < 5: code = "print('error')"

        lang_str = "unknown"
        cols_lower = {c.lower(): c for c in self.data.columns}
        if "language" in cols_lower:
            lang_col = cols_lower["language"]
            lang_str = str(self.data.at[idx, lang_col]).lower().strip()
        
        lang_id = self.lang_map.get(lang_str, self.lang_map.get('unknown', 7))

        enc = self.tokenizer(
            code, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )

        stylo = self.stylo_extractor.extract(code) if self.stylo_extractor else np.zeros(13, dtype=np.float32)

        return {
            "id": row_id,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "stylo_feats": torch.tensor(stylo, dtype=torch.float32),
            "lang_ids": torch.tensor(lang_id, dtype=torch.long)
        }

# -----------------------------------------------------------------------------
# 2. Model Loading
# -----------------------------------------------------------------------------
def load_model_for_submission(config_path: str, checkpoint_dir: str, device: torch.device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Initializing FusionCodeClassifier...")
    model = FusionCodeClassifier(config)
    
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    logger.info(f"Loading weights from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 3. Inference Loop
# -----------------------------------------------------------------------------
def run_inference_pipeline(
    model: FusionCodeClassifier, 
    test_df: pd.DataFrame, 
    id_col_name: str,
    output_file: str, 
    device: torch.device,
    batch_size: int = 64
):
    dataset = SubmissionDataset(test_df, model.tokenizer, max_length=512, id_col=id_col_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ids = []
    predictions = []
    
    THRESHOLD = 0.5 

    logger.info(f"Starting inference on {len(dataset)} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stylo_feats = batch["stylo_feats"].to(device)
            
            with autocast('cuda', dtype=torch.float16):
                logits, _ = model(input_ids, attention_mask, stylo_feats)
                
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= THRESHOLD).long().cpu().tolist()
            
            batch_ids = batch["id"]
            
            if isinstance(batch_ids, torch.Tensor):
                batch_ids = batch_ids.tolist()
            
            elif isinstance(batch_ids, tuple):
                batch_ids = list(batch_ids)
            
            ids.extend(batch_ids)
            predictions.extend(preds)

    # Creazione CSV finale
    submission_df = pd.DataFrame({
        "ID": ids,  
        "label": predictions
    })
    
    output_dir = os.path.dirname(output_file)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    submission_df.to_csv(output_file, index=False)
    logger.info(f"Submission saved to: {output_file}")
    
    # Anteprima
    print("\nPreview:")
    print(submission_df.head().to_string(index=False))

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    config_path = "src/src_TaskA/config/config.yaml"
    checkpoint_dir = "results/results_TaskA/checkpoints"
    
    # --- GESTIONE PERCORSO FILE TEST ---
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = "data/Task_A/test.parquet"
        if not os.path.exists(test_path):
             # Fallback
             test_path = "data/Task_A/test_sample.parquet"

    if not os.path.exists(test_path):
        logger.error(f"Test file not found at {test_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model_for_submission(config_path, checkpoint_dir, device)

    logger.info(f"Reading test data from {test_path}...")
    try:
        df = pd.read_parquet(test_path)
    except:
        df = pd.read_csv(test_path)
        
    # --- RILEVAMENTO COLONNA ID ---
    id_col = None
    for candidate in ["ID", "id", "Id", "sample_id"]:
        if candidate in df.columns:
            id_col = candidate
            break
            
    if id_col is None:
        logger.warning("No ID column found! Generating fake IDs (0..N).")
        df["ID"] = range(len(df))
        id_col = "ID"
    else:
        logger.info(f"Found ID column: '{id_col}'")

    # --- RILEVAMENTO LINGUA ---
    lang_col = None
    for candidate in ["language", "lang", "Language"]:
        if candidate in df.columns:
            lang_col = candidate
            break
    
    if lang_col is None:
        logger.warning("Language column missing. Defaulting to 'unknown'.")
        df["language"] = "unknown"
    else:
        # Normalizziamo il nome per il dataset
        df = df.rename(columns={lang_col: "language"})

    output_file = "results/results_TaskA/submission/submission_task_a.csv"
    
    # Passiamo il nome della colonna ID trovata
    run_inference_pipeline(model, df, id_col, output_file, device)

if __name__ == "__main__":
    main()