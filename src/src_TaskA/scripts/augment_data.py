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

# --- CONFIGURAZIONE VELOCITÀ ---
MODEL_NAME = "microsoft/phi-2"
MAX_SOURCE_LEN = 512
MAX_NEW_TOKENS = 256
BATCH_SIZE = 32

def get_model():
    logger.info(f"Loading Model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.padding_side = "left" 
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16, 
            attn_implementation="sdpa", # Flash Attention nativa
            device_map="cuda",
            trust_remote_code=True
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

def clean_generation(generated_text):
    """
    Estrae solo il codice dopo il marker di Output.
    """
    if "Output:" in generated_text:
        code_part = generated_text.split("Output:")[-1]
    else:
        code_part = generated_text

    # Pulizia Markdown
    code_part = re.sub(r"```[a-zA-Z]*", "", code_part).replace("```", "").strip()
    return code_part

def is_valid_code(code_str):
    if len(code_str) < 10: return False
    # Check simboli base
    if not any(x in code_str for x in ['{', '}', '(', ')', '=', 'def ', 'func', 'import', ';', 'return']):
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/Task_A/train.parquet")
    parser.add_argument("--output_path", type=str, default="data/Task_A/train_augmented.parquet")
    parser.add_argument("--num_samples", type=int, default=3000)
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        return

    logger.info("Loading dataset...")
    df = pd.read_parquet(args.input_path)
    if 'language' in df.columns:
        df['language'] = df['language'].astype(str).str.lower().str.strip()
    
    source_df = df[(df['label'] == 0) & (df['language'] == 'python')].copy()
    
    source_df['char_len'] = source_df['code'].str.len()
    # Scartiamo codici troppo lunghi (fanno perdere tempo) o troppo corti
    source_df = source_df[(source_df['char_len'] > 20) & (source_df['char_len'] < 800)]
    
    # Campioniamo PRIMA di ordinare per avere varietà
    if len(source_df) > args.num_samples:
        source_df = source_df.sample(n=args.num_samples, random_state=42)
    
    # ORA ordiniamo: questo velocizza l'inferenza del 200%
    source_df = source_df.sort_values('char_len').reset_index(drop=True)
    
    logger.info(f"Selected {len(source_df)} samples (Sorted by length).")
    
    model, tokenizer = get_model()
    target_langs = ["go", "c#", "javascript", "php", "c"]
    
    augmented_rows = []
    
    # Loop
    for i in tqdm(range(0, len(source_df), BATCH_SIZE), desc="Fast Translation"):
        batch_df = source_df.iloc[i : i + BATCH_SIZE]
        
        prompts = []
        target_langs_batch = []
        original_codes = []
        
        for idx, row in batch_df.iterrows():
            code = row['code']
            t_lang = target_langs[idx % len(target_langs)]
            
            # Prompt Instruct per Phi-2
            prompt = f"Instruct: Translate this Python code to {t_lang}.\n{code}\nOutput:"
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
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Tagliamo l'input
            generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
            decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for j, full_output in enumerate(decoded_outputs):
                clean_code = clean_generation(full_output)
                
                if is_valid_code(clean_code):
                    augmented_rows.append({
                        "original_code": original_codes[j],
                        "augmented_code": clean_code,
                        "label": 0,
                        "original_lang": "python",
                        "augmented_lang": target_langs_batch[j].lower()
                    })

        except Exception as e:
            logger.error(f"Error: {e}")
            torch.cuda.empty_cache()
            continue

    logger.info(f"Augmentation finished. Generated {len(augmented_rows)} pairs.")

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        # Rinomina per compatibilità
        aug_df = aug_df.rename(columns={
            'original_code': 'code', 
            'augmented_code': 'aug_code',
            'original_lang': 'language'
        })
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        aug_df.to_parquet(args.output_path)
        logger.info(f"Saved clean dataset to {args.output_path}")

if __name__ == "__main__":
    main()