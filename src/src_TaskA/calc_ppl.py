import os
import torch
import argparse
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("PPL_Calculator")

class PerplexityCalculator:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-1.5B", device="cuda"):
        self.device = device
        
        logger.info(f"Loading Tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        logger.info(f"Loading Model: {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()
        
        self.max_length = 2048 
        self.stride = 1024 

    def calculate(self, text: str) -> float:
        """
        Calcola la Cross-Entropy media (Loss).
        Bassa Loss = Il modello riconosce il codice come 'probabile' (spesso AI generated).
        Alta Loss = Il modello è sorpreso (spesso codice Umano idiosincratico).
        """
        # Gestione input sporchi
        if not isinstance(text, str) or len(text.strip()) == 0:
            return np.nan

        encodings = self.tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        if seq_len < 2:
            return np.nan

        nlls = []
        prev_end_loc = 0
        
        # Caso 1: Testo corto
        if seq_len <= self.max_length:
            input_ids = encodings.input_ids.to(self.device)
            target_ids = input_ids.clone()
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                return outputs.loss.item()

        # Caso 2: Sliding Window per testi lunghi
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            
            if begin_loc > 0:
                target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if prev_end_loc == 0: 
            return np.nan
        
        total_nll = torch.stack(nlls).sum() / prev_end_loc
        
        return total_nll.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/Task_A") 
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")
    
    calculator = PerplexityCalculator(args.model_name, device)

    files = ["train.parquet", "validation.parquet", "test.parquet", "test_sample.parquet"]
    
    for filename in files:
        input_path = os.path.join(args.data_dir, filename)
        
        if not os.path.exists(input_path):
            logger.warning(f"File not found: {input_path} -> Skipping.")
            continue
        
        logger.info(f"Processing {filename}...")
        df = pd.read_parquet(input_path)
        
        text_col = next((c for c in ['text', 'code', 'content'] if c in df.columns), None)
        
        if not text_col:
            logger.warning(f"Skipping {filename}: Text column not found.")
            continue

        texts = df[text_col].astype(str).tolist()
        results = []
        
        logger.info(f"Calculating Cross-Entropy for {len(texts)} samples...")
        for t in tqdm(texts):
            results.append(calculator.calculate(t))
            
        df['cross_entropy_loss'] = results
        df['perplexity'] = np.exp(results)

        output_filename = filename.replace(".parquet", "_ppl.parquet")
        output_path = os.path.join(args.data_dir, output_filename)
        
        df.to_parquet(output_path)
        logger.info(f"Saved: {output_path}")
        
        valid_results = [r for r in results if not np.isnan(r)]
        if valid_results:
            mean_ce = np.mean(valid_results)
            logger.info(f"Mean Cross-Entropy for {filename}: {mean_ce:.4f}")
        else:
            logger.warning("No valid results computed.")

if __name__ == "__main__":
    main()