import torch
import pandas as pd
import numpy as np
import zlib
import argparse
import os
import gc
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SemEval_FeatureExtractor")

def calculate_zlib_ratio(text: str) -> float:
    """
    Calcola la compression ratio (Zlib).
    Ratio bassa (< 0.3) = Alta compressione -> Testo molto ripetitivo/prevedibile.
    Ratio alta (> 0.5) = Bassa compressione -> Testo ad alta entropia.
    """
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    
    text_bytes = text.encode('utf-8')
    if len(text_bytes) == 0: return 0.0
    
    compressed_data = zlib.compress(text_bytes)
    return len(compressed_data) / len(text_bytes)

def calculate_nll_sliding_window(
    model, 
    tokenizer, 
    text: str, 
    device: str, 
    max_length: int = 2048, 
    stride: int = 1024
) -> float:
    """
    Calcola la Negative Log-Likelihood (NLL) media con sliding window.
    Ottimizzato per non caricare l'intero tensore in VRAM subito.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.nan

    # 1. Tokenizzazione
    encodings = tokenizer(text, return_tensors="pt")
    input_ids_full = encodings.input_ids
    seq_len = input_ids_full.size(1)

    if seq_len == 0:
        return np.nan

    nlls = []
    prev_end_loc = 0
    
    # 2. Loop a finestra scorrevole
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc 
        
        input_ids = input_ids_full[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        
        if begin_loc > 0:
            context_len = input_ids.size(1) - trg_len
            if context_len > 0:
                target_ids[:, :context_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            
            if not torch.isnan(outputs.loss):
                nlls.append(outputs.loss.item() * trg_len)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    if not nlls:
        return np.nan

    # Media pesata corretta: Somma Loss Totale / Numero Token Totali
    return sum(nlls) / seq_len

def main():
    parser = argparse.ArgumentParser(description="SemEval 2026 Task 13: Feature Extractor")
    parser.add_argument("--data_dir", type=str, default="data/Task_A", help="Cartella contenente i file parquet")
    parser.add_argument("--files", nargs="+", default=["train.parquet", "validation.parquet", "test.parquet"], help="Lista file da processare")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B", help="Modello HF")
    parser.add_argument("--max_length", type=int, default=2048, help="Context Window GPU")
    parser.add_argument("--stride", type=int, default=1024, help="Stride Window")
    args = parser.parse_args()

    # --- SETUP GPU OTTIMIZZATO ---
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True 
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = "cpu"
    
    logger.info(f"Device: {device} | TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
    logger.info(f"Model: {args.model_name}")

    # Caricamento Modello
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.eval()
    except Exception as e:
        logger.error(f"Errore caricamento modello: {e}")
        return

    # Processamento File
    for filename in args.files:
        input_path = os.path.join(args.data_dir, filename)
        if not os.path.exists(input_path):
            logger.warning(f"File non trovato: {input_path}, skip.")
            continue
        
        output_path = os.path.join(args.data_dir, filename.replace(".parquet", "_features.parquet"))
        if os.path.exists(output_path):
            logger.warning(f"File output già esistente: {output_path}. Sovrascrivo...")

        logger.info(f"--- Processing {filename} ---")
        df = pd.read_parquet(input_path)
        
        text_col = next((c for c in ['text', 'code', 'content'] if c in df.columns), None)
        if not text_col:
            logger.error(f"Colonna testo non trovata in {filename}. Skip.")
            continue

        nll_values = []
        zlib_values = []
        
        # tqdm per monitorare progresso
        pbar = tqdm(total=len(df), desc=f"Processing {filename}")
        
        texts = df[text_col].astype(str).tolist()

        for idx, text in enumerate(texts):
            # 1. Calcolo NLL
            try:
                loss = calculate_nll_sliding_window(
                    model, tokenizer, text, device, 
                    max_length=args.max_length, 
                    stride=args.stride
                )
            except RuntimeError as e:
                # Gestione Avanzata OOM (Out of Memory)
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM detectato su sample {idx}. Provo recovery con contesto ridotto...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    try:
                        # Fallback con contesto dimezzato
                        loss = calculate_nll_sliding_window(
                            model, tokenizer, text, device, 
                            max_length=1024, stride=512
                        )
                    except:
                        loss = np.nan
                else:
                    logger.error(f"Errore runtime su sample {idx}: {e}")
                    loss = np.nan
            except Exception as e:
                loss = np.nan

            # 2. Calcolo Zlib
            try:
                z_ratio = calculate_zlib_ratio(text)
            except:
                z_ratio = 0.0

            nll_values.append(loss)
            zlib_values.append(z_ratio)
            pbar.update(1)

            if idx % 1000 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        pbar.close()

        # Salvataggio
        df['nll_loss'] = nll_values
        df['zlib_ratio'] = zlib_values
        
        # Salviamo in float32 per risparmiare spazio su disco se il dataset è enorme
        df['nll_loss'] = df['nll_loss'].astype("float32")
        df['zlib_ratio'] = df['zlib_ratio'].astype("float32")

        logger.info(f"Salvataggio in: {output_path}")
        df.to_parquet(output_path)
        
        # Cleanup finale per il prossimo file
        del df, texts, nll_values, zlib_values
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Pipeline completata.")

if __name__ == "__main__":
    main()