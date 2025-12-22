import os
import re
import torch
import pandas as pd
import logging
import argparse
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configurazione Costanti
MODEL_NAME = "bigcode/starcoder2-3b"
MAX_SOURCE_LEN = 1024
MAX_NEW_TOKENS = 512

def get_model():
    logger.info(f"Loading Model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # StarCoder non ha un pad token di default, usiamo EOS
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

def clean_generation(generated_text, target_lang):
    """
    Pulisce l'output del modello rimuovendo markdown e prompt residui.
    """
    # 1. Rimuovi blocchi markdown (```go ... ```)
    pattern = r"```(?:{})?(.*?)```".format(target_lang.lower())
    match = re.search(pattern, generated_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 2. Fallback: Cerca di tagliare dopo il prompt se non ha usato markdown
    # Spesso il modello ripete il commento "# Go:"
    marker = f"# {target_lang}:"
    if marker in generated_text:
        return generated_text.split(marker)[-1].strip()
        
    return generated_text.strip()

def translate_code(model, tokenizer, code, target_lang):
    """
    Traduce lo snippet preservando la logica. Include gestione errori e troncamento.
    """
    # Troncamento preventivo per evitare errori di context length
    if len(code) > MAX_SOURCE_LEN * 4: # Stima caratteri -> token
        code = code[:MAX_SOURCE_LEN * 4]

    # Prompt ottimizzato per StarCoder2
    prompt = f"<filename>solution.{target_lang.lower()}\n# Translate the following Python code to {target_lang}.\n# Python:\n{code}\n\n# {target_lang}:\n"
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        terminators = [tokenizer.eos_token_id]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
                repetition_penalty=1.1
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_only = full_output[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        
        return clean_generation(gen_only, target_lang)

    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/Task_A/train.parquet")
    parser.add_argument("--output_path", type=str, default="data/Task_A/train_augmented.parquet")
    parser.add_argument("--num_samples", type=int, default=3000, help="Samples to translate")
    parser.add_argument("--batch_size", type=int, default=1, help="Keep 1 for stability with LLMs")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        logger.error(f"Input file not found: {args.input_path}")
        return

    logger.info("Loading dataset...")
    df = pd.read_parquet(args.input_path)
    
    # Filtro: Solo Human (0) e Solo Python
    source_df = df[(df['label'] == 0) & (df['language'] == 'python')].copy()
    
    if source_df.empty:
        logger.error("No valid source data found (Python + Human).")
        return

    # Campionamento se necessario
    if len(source_df) > args.num_samples:
        source_df = source_df.sample(n=args.num_samples, random_state=42)
    
    logger.info(f"Selected {len(source_df)} samples for augmentation.")
    
    model, tokenizer = get_model()
    
    target_langs = ["go", "c#", "javascript", "php", "c"]
    
    augmented_rows = []
    success_count = 0
    
    logger.info("Starting translation loop...")
    for idx, row in tqdm(source_df.iterrows(), total=len(source_df), dynamic_ncols=True):
        original_code = row['code']
        
        # Controllo validità codice sorgente
        if not isinstance(original_code, str) or len(original_code.strip()) < 10:
            continue
            
        target_lang = target_langs[idx % len(target_langs)]
        
        trans_code = translate_code(model, tokenizer, original_code, target_lang)
        
        if trans_code and len(trans_code.strip()) > 10:
            augmented_rows.append({
                "original_code": original_code,
                "augmented_code": trans_code,
                "label": 0, 
                "original_lang": "python",
                "augmented_lang": target_lang.lower()
            })
            success_count += 1
        
        if idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    logger.info(f"Augmentation finished. Success rate: {success_count}/{len(source_df)}")

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        aug_df.to_parquet(args.output_path)
        logger.info(f"Saved paired dataset to {args.output_path}")
    else:
        logger.warning("No data generated.")

if __name__ == "__main__":
    main()