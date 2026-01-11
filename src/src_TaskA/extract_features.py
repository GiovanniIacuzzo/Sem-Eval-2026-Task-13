import os
import sys
import torch
import argparse
import logging
import pandas as pd
import math
import re
from tqdm import tqdm
from typing import Dict, List
from collections import Counter
from transformers import AutoTokenizer, AutoModel

# --- CONFIGURAZIONE LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FeatureExtractor")

# --- 1. STYLOMETRIC ENGINE ---
class StylometricEngine:
    """
    Motore per l'estrazione di feature sintattiche e stilistiche.
    Combina Perplexity (dal LLM) con metriche classiche (Entropia, TTR, ecc.)
    """
    def __init__(self):
        # Regex pre-compilate per efficienza
        self.token_pattern = re.compile(r'\b\w+\b')
        self.comment_pattern = re.compile(r'(#|//|/\*|\*)') 

    def _calculate_entropy(self, text: str) -> float:
        """Calcola l'entropia di Shannon dei caratteri."""
        if not text: return 0.0
        counts = Counter(text)
        length = len(text)
        probs = [count / length for count in counts.values()]
        return -sum(p * math.log(p, 2) for p in probs)

    def process(self, code: str, perplexity: float) -> List[float]:
        """
        Input: 
            code (str): Snippet di codice
            perplexity (float): Valore pre-calcolato (o 0.0)
        Output: 
            List[float]: Vettore delle feature manuali
        """
        if not isinstance(code, str) or len(code.strip()) == 0:
            # Ritorna vettore di zeri se il codice è vuoto
            return [0.0] * 9 

        # A. Statistiche base
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        num_lines = len(lines)
        num_chars = len(code)
        
        # B. Lexical Diversity (Type-Token Ratio)
        tokens = self.token_pattern.findall(code)
        num_tokens = len(tokens)
        ttr = len(set(tokens)) / num_tokens if num_tokens > 0 else 0.0

        # C. Densità Commenti (Euristica)
        comment_lines = sum(1 for l in non_empty_lines if self.comment_pattern.match(l.strip()))
        comment_density = comment_lines / len(non_empty_lines) if non_empty_lines else 0.0

        # D. Whitespace Ratio (Spazi bianchi sul totale caratteri)
        whitespace_count = sum(1 for c in code if c.isspace())
        whitespace_ratio = whitespace_count / num_chars if num_chars > 0 else 0.0

        # E. Lunghezza Linee
        line_lengths = [len(l) for l in lines]
        avg_line_len = sum(line_lengths) / num_lines if num_lines > 0 else 0.0
        max_line_len = max(line_lengths) if line_lengths else 0.0

        # F. Entropia
        entropy = self._calculate_entropy(code)

        # G. Snake_case ratio (underscore density)
        underscore_density = code.count('_') / num_chars if num_chars > 0 else 0.0
        
        # H. CamelCase proxy (maiuscole non all'inizio)
        upper_density = sum(1 for c in code if c.isupper()) / num_chars if num_chars > 0 else 0.0

        # COSTRUZIONE VETTORE FEATURES (Totale: 9 Features)
        features = [
            perplexity,         # 1. AI Surprise
            ttr,                # 2. Ripetitività lessicale
            comment_density,    # 3. Presenza di commenti
            whitespace_ratio,   # 4. Uso dello spazio
            avg_line_len,       # 5. Lunghezza media
            max_line_len,       # 6. Lunghezza massima (proxy per minified code)
            entropy,            # 7. Complessità caratteri
            underscore_density, # 8. Stile (snake_case)
            upper_density       # 9. Stile (CamelCase)
        ]
        
        return features

# --- 2. VECTORIZER ---
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
            logger.warning("Colonna 'perplexity' ASSENTE! Imposto a 0.0.")
            ppls = [0.0] * len(df)
        else:
            ppls = df['perplexity'].fillna(0.0).astype(float).tolist()

        codes = df['code'].astype(str).tolist()
        num_samples = len(codes)

        # --- FASE 1: Estrazione Feature Stilometriche (CPU) ---
        logger.info(f"Extracting Stylometric Features (PPL + Style) for {num_samples} samples...")
        manual_features_list = []
        
        for code, ppl in tqdm(zip(codes, ppls), total=num_samples, desc="Style Extraction"):
            feats = self.style_engine.process(code, ppl)
            manual_features_list.append(feats)
            
        manual_tensor = torch.tensor(manual_features_list, dtype=torch.float32)

        # --- FASE 2: Estrazione Embedding Semantici (GPU - UniXcoder) ---
        logger.info(f"Extracting Semantic Embeddings (UniXcoder) in batches of {batch_size}...")
        embedding_list = []
        
        for i in tqdm(range(0, num_samples, batch_size), desc="Semantic Extraction"):
            batch_codes = codes[i : i + batch_size]
            
            inputs = self.tokenizer(
                batch_codes, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean Pooling per ottenere [Batch, 768]
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
                
            embedding_list.append(embeddings)
        
        embedding_tensor = torch.cat(embedding_list, dim=0)

        # --- FASE 3: Output Finale ---
        labels_tensor = torch.tensor(df['label'].values, dtype=torch.long)
        
        logger.info(f"Extraction Complete. Shapes -> Emb: {embedding_tensor.shape}, Feat: {manual_tensor.shape}")
        
        return {
            "embeddings": embedding_tensor,      # [N, 768] (UniXcoder)
            "features": manual_tensor,           # [N, 9]   (Style + PPL)
            "labels": labels_tensor,             # [N]
            "languages": df['language'].values,  # Numpy Array
            "ids": df.index.values               # Utile per tracciare errori
        }

# --- 3. MAIN ---
def main():
    parser = argparse.ArgumentParser(description="SemEval Task A - Feature Extraction")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing parquet files with 'perplexity'")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save vectorized .pt files")
    parser.add_argument("--model_name", type=str, default="microsoft/unixcoder-base")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Istanziamo il Vectorizer
    vectorizer = Vectorizer(args.model_name, device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Definisci i file da cercare. 
    files_map = {
        "train": "train.parquet",
        "val": "validation.parquet",
        "test": "test.parquet"
    }

    processed_count = 0

    available_files = [f for f in os.listdir(args.data_dir) if f.endswith('.parquet')]
    logger.info(f"Found files in {args.data_dir}: {available_files}")

    for split_name, target_filename in files_map.items():
        # Cerchiamo il file
        found_path = None
        possible_names = [target_filename, target_filename.replace(".parquet", "_ppl.parquet")]
        
        for name in possible_names:
            path = os.path.join(args.data_dir, name)
            if os.path.exists(path):
                found_path = path
                break
        
        if not found_path:
            logger.warning(f"Skipping split '{split_name}': Could not find {possible_names}")
            continue

        logger.info(f"Processing split: {split_name.upper()} from {found_path}")
        
        try:
            df = pd.read_parquet(found_path)
            
            if len(df) == 0:
                logger.warning(f"Dataset {split_name} is empty!")
                continue
            
            # Processing
            data_dict = vectorizer.process_dataset(df, batch_size=args.batch_size)
            
            # Saving
            output_file = os.path.join(args.output_dir, f"{split_name}_vectors.pt")
            torch.save(data_dict, output_file)
            logger.info(f"Saved vectorized data to {output_file}")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Critical error processing {split_name}: {e}", exc_info=True)

    if processed_count == 0:
        logger.error("No files were processed. Check your file names and paths.")
    else:
        logger.info("All tasks completed successfully.")

if __name__ == "__main__":
    main()