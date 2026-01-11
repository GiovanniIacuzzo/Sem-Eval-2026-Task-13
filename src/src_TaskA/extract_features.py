import os
import sys
import torch
import argparse
import logging
import pandas as pd
import numpy as np
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
    Motore aggiornato con le euristiche "Code Forensics" suggerite dal professore.
    """
    def __init__(self):
        # 1. Regex Generiche
        self.token_pattern = re.compile(r'\b\w+\b')
        self.comment_pattern = re.compile(r'(#|//|/\*|\*)')
        
        # 2. Regex per Identificatori
        self.id_pattern = re.compile(r'\b[a-zA-Z_]\w*\b')
        
        # 3. Regex per Stile Naming
        self.snake_case = re.compile(r'\b[a-z]+(_[a-z]+)+\b')
        self.camel_case = re.compile(r'\b[a-z]+([A-Z][a-z0-9]*)+\b')
        
        # 4. Regex per Spaziatura Operatori (Inconsistenza)
        self.op_tight = re.compile(r'\w+=\w+')
        self.op_loose = re.compile(r'\w+ = \w+')

        # 5. Regex Commenti "Sospetti" (Umani)
        self.suspicious_comment = re.compile(r'(TODO|FIXME|\?\?\?|hack|broken|temp)', re.IGNORECASE)

    def _calculate_entropy(self, text: str) -> float:
        """Calcola l'entropia di Shannon."""
        if not text: return 0.0
        counts = Counter(text)
        length = len(text)
        probs = [count / length for count in counts.values()]
        return -sum(p * math.log(p, 2) for p in probs)

    def process(self, code: str, perplexity: float) -> List[float]:
        if not isinstance(code, str) or len(code.strip()) == 0:
            return [0.0] * 16 # Aggiornato a 16 features totali

        # --- A. PRE-PROCESSING ---
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        num_lines = len(lines)
        num_chars = len(code)
        
        # Estrazione token e identificatori
        tokens = self.token_pattern.findall(code)
        identifiers = self.id_pattern.findall(code)
        num_tokens = len(tokens)
        num_ids = len(identifiers)

        # --- B. FEATURES BASE ---
        
        # 1. Perplexity (dal LLM precedente)
        feat_ppl = perplexity
        
        # 2. Lexical Diversity (Type-Token Ratio)
        feat_ttr = len(set(tokens)) / num_tokens if num_tokens > 0 else 0.0

        # 3. Densità Commenti
        comment_lines = sum(1 for l in non_empty_lines if self.comment_pattern.match(l.strip()))
        feat_comment_density = comment_lines / len(non_empty_lines) if non_empty_lines else 0.0

        # 4. Whitespace Ratio
        feat_whitespace_ratio = sum(1 for c in code if c.isspace()) / num_chars if num_chars > 0 else 0.0

        # 5. Line Length Mean
        line_lengths = [len(l) for l in lines]
        feat_avg_line_len = np.mean(line_lengths) if line_lengths else 0.0

        # 6. Line Length Std Dev
        feat_std_line_len = np.std(line_lengths) if line_lengths else 0.0

        # 7. Global Character Entropy
        feat_entropy = self._calculate_entropy(code)

        # --- C. FEATURES IDENTIFICATORI ---

        if num_ids > 0:
            id_lengths = [len(x) for x in identifiers]
            
            # 8. Avg Identifier Length
            feat_id_avg_len = np.mean(id_lengths)
            
            # 9. Short Identifier Ratio
            feat_id_short_ratio = sum(1 for x in id_lengths if x < 3) / num_ids
            
            # 10. Numeric in Identifiers
            feat_id_numeric_ratio = sum(1 for x in identifiers if any(c.isdigit() for c in x)) / num_ids
            
            # 11. Identifier Character Entropy
            all_ids_str = "".join(identifiers)
            feat_id_char_entropy = self._calculate_entropy(all_ids_str)
        else:
            feat_id_avg_len = 0.0
            feat_id_short_ratio = 0.0
            feat_id_numeric_ratio = 0.0
            feat_id_char_entropy = 0.0

        # --- D. INCONSISTENZE & STILE ---

        # 12. Case Mixing Score
        snake_count = len(self.snake_case.findall(code))
        camel_count = len(self.camel_case.findall(code))
        total_style_cases = snake_count + camel_count
        if total_style_cases > 5:
            minority = min(snake_count, camel_count)
            feat_case_mix = (minority / total_style_cases) * 2 
        else:
            feat_case_mix = 0.0

        # 13. Spacing Inconsistency
        tight_ops = len(self.op_tight.findall(code))
        loose_ops = len(self.op_loose.findall(code))
        total_ops = tight_ops + loose_ops
        if total_ops > 2:
            feat_spacing_incons = (min(tight_ops, loose_ops) / total_ops) * 2
        else:
            feat_spacing_incons = 0.0

        # 14. Underscore Density (Classico)
        feat_underscore_density = code.count('_') / num_chars if num_chars > 0 else 0.0
        
        # 15. Suspicious Comments
        feat_suspicious_comments = len(self.suspicious_comment.findall(code))

        # 16. Max Line Length
        feat_max_line_len = max(line_lengths) if line_lengths else 0.0

        # COSTRUZIONE VETTORE FEATURES (Totale: 16 Features)
        features = [
            feat_ppl,                # 1. AI Surprise (Loss)
            feat_ttr,                # 2. Ripetitività Lessicale
            feat_comment_density,    # 3. Densità commenti
            feat_whitespace_ratio,   # 4. Uso spazio
            feat_avg_line_len,       # 5. Lunghezza media riga
            feat_std_line_len,       # 6. Irregolarità lunghezza righe
            feat_entropy,            # 7. Complessità globale
            feat_id_avg_len,         # 8. Lunghezza media variabili
            feat_id_short_ratio,     # 9. % variabili corte
            feat_id_numeric_ratio,   # 10. % variabili con numeri
            feat_id_char_entropy,    # 11. Entropia interna variabi
            feat_case_mix,           # 12. Inconsistenza camel/snake
            feat_spacing_incons,     # 13. Inconsistenza spazi operatori
            feat_underscore_density, # 14. Densità underscore
            feat_suspicious_comments,# 15. Commenti "umani"
            feat_max_line_len        # 16. Max Line Length
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
        
        logger.info("Initializing Stylometric Engine (Forensics Mode)...")
        self.style_engine = StylometricEngine()

    def process_dataset(self, df: pd.DataFrame, batch_size: int = 64) -> Dict[str, torch.Tensor]:
        required_cols = ['code', 'label']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Gestione Loss/Perplexity
        ppl_col = next((c for c in ['cross_entropy_loss', 'perplexity', 'loss'] if c in df.columns), None)
        if not ppl_col:
            logger.warning("Colonna Perplexity/Loss ASSENTE! Uso 0.0 come default.")
            ppls = [0.0] * len(df)
        else:
            logger.info(f"Using column '{ppl_col}' for AI-Surprise feature.")
            ppls = df[ppl_col].fillna(0.0).astype(float).tolist()

        codes = df['code'].astype(str).tolist()
        num_samples = len(codes)

        # --- FASE 1: Estrazione Feature Stilometriche (CPU) ---
        logger.info(f"Extracting Stylometric Features (16 metrics) for {num_samples} samples...")
        manual_features_list = []
        
        for code, ppl in tqdm(zip(codes, ppls), total=num_samples, desc="Forensics Extraction"):
            feats = self.style_engine.process(code, ppl)
            manual_features_list.append(feats)
            
        manual_tensor = torch.tensor(manual_features_list, dtype=torch.float32)

        # --- FASE 2: Estrazione Embedding Semantici (GPU - UniXcoder) ---
        logger.info(f"Extracting Semantic Embeddings (UniXcoder) in batches of {batch_size}...")
        embedding_list = []
        
        for i in tqdm(range(0, num_samples, batch_size), desc="Semantic Extraction"):
            batch_codes = codes[i : i + batch_size]
            
            # Tronchiamo a 512 token
            inputs = self.tokenizer(
                batch_codes, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512 
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
                
            embedding_list.append(embeddings)
        
        embedding_tensor = torch.cat(embedding_list, dim=0)

        # --- FASE 3: Output Finale ---
        labels_tensor = torch.tensor(df['label'].values, dtype=torch.long)
        
        # Gestione opzionale colonna language
        langs = df['language'].values if 'language' in df.columns else np.array(['unknown']*num_samples)

        logger.info(f"Complete. Emb Shape: {embedding_tensor.shape}, Manual Feat Shape: {manual_tensor.shape}")
        
        return {
            "embeddings": embedding_tensor,      # [N, 768] (UniXcoder)
            "features": manual_tensor,           # [N, 16]  (Forensics Features)
            "labels": labels_tensor,             # [N]
            "languages": langs,                  # Numpy Array
            "ids": df.index.values               # Metadati
        }

# --- 3. MAIN ---
def main():
    parser = argparse.ArgumentParser(description="SemEval Task A - Feature Extraction")
    parser.add_argument("--data_dir", type=str, default="data/Task_A", help="Dir con file _ppl.parquet")
    parser.add_argument("--output_dir", type=str, default="data/vectors", help="Output .pt files")
    parser.add_argument("--model_name", type=str, default="microsoft/unixcoder-base")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vectorizer = Vectorizer(args.model_name, device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    files_map = {
        "train": "train.parquet",
        "val": "validation.parquet",
        "test": "test.parquet",
        "test_sample": "test_sample.parquet"
    }

    processed_count = 0

    for split_name, base_filename in files_map.items():
        ppl_filename = base_filename.replace(".parquet", "_ppl.parquet")
        
        path_ppl = os.path.join(args.data_dir, ppl_filename)
        path_base = os.path.join(args.data_dir, base_filename)
        
        if os.path.exists(path_ppl):
            input_path = path_ppl
        elif os.path.exists(path_base):
            input_path = path_base
            logger.warning(f"File _ppl non trovato per {split_name}. Uso {base_filename} (Perplexity sarà 0).")
        else:
            logger.warning(f"Skipping {split_name}: File not found.")
            continue

        logger.info(f"Processing {split_name.upper()} from {input_path}")
        
        try:
            df = pd.read_parquet(input_path)
            data_dict = vectorizer.process_dataset(df, batch_size=args.batch_size)
            
            output_file = os.path.join(args.output_dir, f"{split_name}_vectors.pt")
            torch.save(data_dict, output_file)
            logger.info(f"Saved: {output_file}")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {split_name}: {e}", exc_info=True)

    if processed_count > 0:
        logger.info("Extraction Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()