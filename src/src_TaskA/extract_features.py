import os
import sys
import torch
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from typing import Dict
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.src_TaskA.dataset.dataset import StylometricEngine

# Configurazione Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FeatureExtractor")

class Vectorizer:
    def __init__(self, model_name: str, device: str):
        self.device = device
        logger.info(f"Loading Semantic Backbone: {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        logger.info("Initializing Stylometric Engine...")
        self.style_engine = StylometricEngine()

    def process_dataset(self, df: pd.DataFrame, batch_size: int = 64) -> Dict[str, torch.Tensor]:
        """
        Processa un DataFrame completo e restituisce un dizionario di tensori pronti per il training.
        """
        required_cols = ['code', 'label', 'language']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if 'perplexity' not in df.columns:
            logger.warning("Colonna 'perplexity' ASSENTE! Il modello sarà cieco su questa feature. Imposto a 0.0.")
            ppls = [0.0] * len(df)
        else:
            ppls = df['perplexity'].astype(float).tolist()

        codes = df['code'].astype(str).tolist()
        num_samples = len(codes)

        # ---------------------------------------------------------
        # 1. Estrazione Feature Stilometriche
        # ---------------------------------------------------------
        logger.info(f"Extracting Stylometric Features for {num_samples} samples...")
        manual_features_list = []
        
        for code, ppl in tqdm(zip(codes, ppls), total=num_samples, desc="Style Engine"):
            feats = self.style_engine.process(code, ppl)
            manual_features_list.append(feats)
            
        manual_tensor = torch.tensor(manual_features_list, dtype=torch.float32)

        # ---------------------------------------------------------
        # 2. Estrazione Embedding Semantici
        # ---------------------------------------------------------
        logger.info(f"Extracting Semantic Embeddings (UniXcoder) in batches of {batch_size}...")
        embedding_list = []
        
        for i in tqdm(range(0, num_samples, batch_size), desc="UniXcoder"):
            batch_codes = codes[i : i + batch_size]
            
            # Tokenizzazione
            inputs = self.tokenizer(
                batch_codes, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Dimensione: [Batch, 768]
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
                
            embedding_list.append(embeddings)
        
        embedding_tensor = torch.cat(embedding_list, dim=0)

        # ---------------------------------------------------------
        # 3. Assemblaggio Output
        # ---------------------------------------------------------
        labels_tensor = torch.tensor(df['label'].values, dtype=torch.long)
        
        logger.info(f"Extraction Complete. Shapes -> Emb: {embedding_tensor.shape}, Feat: {manual_tensor.shape}")
        
        return {
            "embeddings": embedding_tensor,  # [N, 768]
            "features": manual_tensor,       # [N, 10]
            "labels": labels_tensor,         # [N]
            "languages": df['language'].values, # Array numpy di stringhe (utile per split LOLO)
            "ids": df.index.values           # Utile per debugging
        }

def main():
    parser = argparse.ArgumentParser(description="SemEval Task A - Offline Feature Extraction")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing _ppl_burst.parquet files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save vectorized .pt files")
    parser.add_argument("--model_name", type=str, default="microsoft/unixcoder-base")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vectorizer = Vectorizer(args.model_name, device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Lista dei file da processare
    files_to_process = {
        "train": "train_ppl_burst.parquet",
        "val": "validation_ppl_burst.parquet",
        "test_sample": "test_sample_ppl_burst.parquet"
    }

    for split_name, filename in files_to_process.items():
        file_path = os.path.join(args.data_dir, filename)
        
        if not os.path.exists(file_path):
            # Fallback ai nomi vecchi se necessario
            logger.warning(f"File {filename} not found. Trying fallback without '_burst'...")
            filename = filename.replace("_burst", "")
            file_path = os.path.join(args.data_dir, filename)
            
            if not os.path.exists(file_path):
                logger.error(f"Skipping {split_name}: File not found at {file_path}")
                continue

        logger.info(f"Processing split: {split_name.upper()} from {file_path}")
        try:
            df = pd.read_parquet(file_path)
            
            # Controllo integrità
            if len(df) == 0:
                logger.warning(f"Dataset {split_name} is empty!")
                continue
                
            data_dict = vectorizer.process_dataset(df, batch_size=args.batch_size)
            
            output_path = os.path.join(args.output_dir, f"{split_name}_vectors.pt")
            torch.save(data_dict, output_path)
            logger.info(f"Saved vectorized data to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to process {split_name}: {e}", exc_info=True)

    logger.info("All tasks completed.")

if __name__ == "__main__":
    main()