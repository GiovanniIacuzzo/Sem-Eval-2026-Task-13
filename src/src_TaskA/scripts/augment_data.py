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

# --- CONFIGURAZIONE TURBO ---
MODEL_NAME = "bigcode/starcoder2-3b"
MAX_SOURCE_LEN = 512
MAX_NEW_TOKENS = 384
BATCH_SIZE = 48

def get_model():
    logger.info(f"Loading Model: {MODEL_NAME} (Turbo Mode)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.padding_side = "left" 
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16, 
            attn_implementation="sdpa",
            device_map=None
        )
        model.to("cuda")
        model.eval()
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

def clean_generation(generated_text, target_lang):
    pattern = r"```(?:{})?(.*?)```".format(target_lang.lower())
    match = re.search(pattern, generated_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    marker = f"# {target_lang}:"
    if marker in generated_text:
        return generated_text.split(marker)[-1].strip()
        
    return generated_text.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/Task_A/train.parquet")
    parser.add_argument("--output_path", type=str, default="data/Task_A/train_paired_augmented.parquet")
    parser.add_argument("--num_samples", type=int, default=3000)
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        return

    logger.info("Loading dataset...")
    df = pd.read_parquet(args.input_path)
    if 'language' in df.columns:
        df['language'] = df['language'].astype(str).str.lower().str.strip()
    
    source_df = df[(df['label'] == 0) & (df['language'] == 'python')].copy()
    
    if len(source_df) > args.num_samples:
        source_df = source_df.sample(n=args.num_samples, random_state=42)
    
    source_df = source_df.reset_index(drop=True)
    
    logger.info(f"Selected {len(source_df)} samples. Processing with BATCH_SIZE={BATCH_SIZE}")
    
    model, tokenizer = get_model()
    target_langs = ["go", "c#", "javascript", "php", "c"]
    
    augmented_rows = []
    
    # Loop Principale
    for i in tqdm(range(0, len(source_df), BATCH_SIZE), desc="Turbo Translation"):
        batch_df = source_df.iloc[i : i + BATCH_SIZE]
        
        prompts = []
        target_langs_batch = []
        original_codes = []
        
        for idx, row in batch_df.iterrows():
            code = row['code']
            if len(code) > MAX_SOURCE_LEN * 3: code = code[:MAX_SOURCE_LEN * 3]
            
            t_lang = target_langs[idx % len(target_langs)]
            
            prompt = f"<filename>solution.{t_lang.lower()}\n# Translate the following Python code to {t_lang}.\n# Python:\n{code}\n\n# {t_lang}:\n"
            prompts.append(prompt)
            target_langs_batch.append(t_lang)
            original_codes.append(code)

        if not prompts: continue

        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SOURCE_LEN).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, 
                    temperature=None, 
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for j, full_output in enumerate(decoded_outputs):
                clean_code = clean_generation(full_output, target_langs_batch[j])
                
                if clean_code and len(clean_code) > 10:
                    augmented_rows.append({
                        "original_code": original_codes[j],
                        "augmented_code": clean_code,
                        "label": 0,
                        "original_lang": "python",
                        "augmented_lang": target_langs_batch[j].lower()
                    })

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("OOM detected! Clearing cache and skipping batch. Reduce BATCH_SIZE if frequent.")
                torch.cuda.empty_cache()
                continue
            else:
                logger.error(f"Batch error: {e}")
                continue
        except Exception as e:
            logger.error(f"Generic error: {e}")
            continue
            
        if i % (BATCH_SIZE * 5) == 0:
            torch.cuda.empty_cache()

    logger.info(f"Augmentation finished. Generated {len(augmented_rows)} pairs.")

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        aug_df.to_parquet(args.output_path)
        logger.info(f"Saved paired dataset to {args.output_path}")

if __name__ == "__main__":
    main()