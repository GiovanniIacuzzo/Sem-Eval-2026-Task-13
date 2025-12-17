import os
import sys
import logging
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast
from dotenv import load_dotenv
from typing import List, Optional

# Assicurati che questi import puntino ai tuoi file corretti
from src_TaskB.models.model import CodeClassifier
# Importiamo la classe appena creata
from src_TaskB.dataset.Inference_dataset import InferenceDataset

# PEFT import per gestire LoRA
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

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

class ConsoleUX:
    @staticmethod
    def print_banner(text: str):
        print(f"\n{'-'*60}\n{text.center(60)}\n{'-'*60}")

# -----------------------------------------------------------------------------
# Submission Logic
# -----------------------------------------------------------------------------
def run_inference_pipeline(
    model_wrapper: CodeClassifier, 
    test_df: pd.DataFrame, 
    id_col_name: str, 
    output_file: str, 
    device: torch.device,
    max_length: int,      # <--- FIX: Passiamo max_length esplicitamente
    batch_size: int = 32
):
    model_wrapper.eval()
    
    # Dataset di Inferenza (Speciale per submission, restituisce ID)
    dataset = InferenceDataset(
        dataframe=test_df, 
        tokenizer=model_wrapper.tokenizer, 
        max_length=max_length, # <--- FIX: Usiamo il valore da config
        id_col=id_col_name
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True if device.type == 'cuda' else False
    )

    ids: List[str] = []
    predictions: List[int] = []

    logger.info(f"Starting inference on {len(dataset)} samples...")

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    dtype = torch.float16 if device.type == 'cuda' else torch.float32

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Generating Predictions", dynamic_ncols=True)
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            batch_ids = batch["id"]

            with autocast(device_type=device_type, dtype=dtype):
                # FIX DANN: Passiamo alpha=0.0 e lang_ids=None
                # In fase di submission non ci interessa il language classification
                logits, _ = model_wrapper(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    lang_ids=None,
                    alpha=0.0
                )
            
            # Predizione della classe (0-30)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            
            ids.extend(batch_ids)
            predictions.extend(preds)

    # Creazione DataFrame sottomissione
    submission_df = pd.DataFrame({
        "ID": ids,
        "label": predictions
    })
    
    # Assicurati che la directory di output esista
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    submission_df.to_csv(output_file, index=False)
    
    logger.info(f"✅ Submission saved to: {output_file}")
    print(f"\nPreview:\n{submission_df.head().to_string(index=False)}")

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------
def main():
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    ConsoleUX.print_banner("SemEval Task 13B - Submission Generator")

    # 1. Configurazione
    config_path = "src_TaskB/config/config.yaml"
    if not os.path.exists(config_path):
        logger.critical(f"Config file missing: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # 3. Inizializzazione Modello
    logger.info("Initializing Model Architecture...")
    model_wrapper = CodeClassifier(config)
    model_wrapper.to(device)

    # 4. Caricamento Pesi
    checkpoint_dir = os.path.abspath("results_TaskB/checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        logger.critical(f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    is_peft = config["model"].get("use_lora", True)

    if is_peft:
        logger.info(f"Loading LoRA Adapters form: {checkpoint_dir}")
        try:
            # 1. Carica Adapter LoRA
            model_wrapper.base_model.load_adapter(checkpoint_dir, adapter_name="default")
            model_wrapper.base_model.set_adapter("default")
            
            # 2. Carica Custom Heads
            custom_path = os.path.join(checkpoint_dir, "custom_components.pt")
            if os.path.exists(custom_path):
                custom_state = torch.load(custom_path, map_location=device)
                
                # --- FIX CHIAVI ---
                # Classifier
                model_wrapper.classifier.load_state_dict(custom_state['classifier'])
                
                # Pooler (Controlliamo se la chiave è 'pooler' o 'attention_pooler')
                if 'pooler' in custom_state:
                    model_wrapper.pooler.load_state_dict(custom_state['pooler'])
                elif 'attention_pooler' in custom_state:
                    model_wrapper.pooler.load_state_dict(custom_state['attention_pooler'])
                else:
                    logger.warning("Key for pooler not found in custom_components.pt")

                # Projection Head (DANN) - Importante caricarla per evitare pesi random
                if 'projection_head' in custom_state:
                    model_wrapper.projection_head.load_state_dict(custom_state['projection_head'])

                logger.info("Custom components loaded.")
            else:
                logger.error(f"⚠️ Custom components file missing at {custom_path}")
                sys.exit(1)
                
        except Exception as e:
            logger.critical(f"Failed to load LoRA/Custom weights: {e}")
            sys.exit(1)
    else:
        # Full model loading (fallback)
        full_path = os.path.join(checkpoint_dir, "full_model.bin")
        model_wrapper.load_state_dict(torch.load(full_path, map_location=device))

    # 5. Caricamento Dati Test
    test_path = os.getenv("TEST_DATA_PATH", "data/Task_B/test.parquet") 
    
    if not os.path.exists(test_path):
        logger.error(f"Test dataset not found at: {test_path}")
        sys.exit(1)

    logger.info(f"Loading Test Data: {test_path}")
    try:
        df = pd.read_parquet(test_path)
    except Exception as e:
        logger.critical(f"Parquet read error: {e}")
        sys.exit(1)
    
    # 6. Identificazione colonna ID
    # A volte i dataset usano 'id', 'ID', 'sample_id', o l'indice stesso
    possible_id_cols = ["id", "ID", "sample_id", "test_id"]
    id_col_name = next((col for col in possible_id_cols if col in df.columns), None)
            
    if not id_col_name:
        # Se non c'è colonna ID, usiamo l'indice del dataframe come ID
        logger.warning(f"No ID column found ({possible_id_cols}). Using DataFrame Index as ID.")
        df['id'] = df.index
        id_col_name = 'id'
    
    # 7. Truncation preventiva
    df['code'] = df['code'].str.slice(0, 8192)

    # 8. Esecuzione
    output_file = "./results_TaskB/submission/submission_task_b.csv"
    
    try:
        # FIX: Recuperiamo max_length dalla config
        max_len = config["data"]["max_length"]
        
        run_inference_pipeline(
            model_wrapper=model_wrapper, 
            test_df=df, 
            id_col_name=id_col_name, 
            output_file=output_file, 
            device=device,
            max_length=max_len, # Passiamo il valore corretto
            batch_size=32 
        )
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        sys.exit(1)

    ConsoleUX.print_banner("Process Completed Successfully")

if __name__ == "__main__":
    main()