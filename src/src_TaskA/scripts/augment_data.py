import os
import re
import torch
import pandas as pd
import logging
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- CONFIGURAZIONE ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct" 
MAX_SOURCE_LEN = 512
MAX_NEW_TOKENS = 512
BATCH_SIZE = 32

def get_model():
    logger.info(f"Loading Model: {MODEL_NAME} in 4-bit...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.padding_side = "left" 
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Configurazione per caricare in 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

def clean_code_block(text, lang):
    """
    Cleaner molto più robusto specifico per Qwen/DeepSeek
    """
    # 1. Rimuovi header
    text = re.sub(r'^(Here is|Sure|Certainly|Below is).*?:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # 2. Estrazione Markdown
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    code_body = ""
    if matches:
        code_body = max(matches, key=len).strip()
    else:
        lines = text.split('\n')
        code_lines = []
        for line in lines:
            if line.strip().lower().startswith(("note:", "explanation:", "usage:", "output:", ">>>")):
                continue
            code_lines.append(line)
        code_body = "\n".join(code_lines).strip()

    return code_body

def is_valid_code(code_str, original_len):
    if len(code_str) < 10: return False
    
    ratio = len(code_str) / (original_len + 1)
    if ratio < 0.2 or ratio > 6.0: 
        return False

    common_syntax = ['{', '}', ';', '(', ')', '=', 'return']
    if not any(x in code_str for x in common_syntax):
        return False
        
    if code_str.lower().startswith(("i'm sorry", "as an ai", "cannot translate")):
        return False
        
    return True

def build_prompt_chat(code, target_lang):
    """
    Usa il formato Chat Template specifico di Qwen/Llama3 per forzare l'output.
    """
    system_prompt = "You are a specialized code transpiler. Your ONLY job is to translate Python code into valid, working code in the target language. Do NOT explain. Do NOT chat. Output ONLY the code inside markdown backticks."
    
    user_content = f"Translate this Python snippet into {target_lang}:\n\n```python\n{code}\n```"
    
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n```"
    
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/Task_A/train.parquet")
    parser.add_argument("--output_path", type=str, default="data/Task_A/train_augmented.parquet")
    parser.add_argument("--num_samples", type=int, default=10000) 
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        logger.error("Input file not found.")
        return

    logger.info("Loading dataset...")
    df = pd.read_parquet(args.input_path)
    if 'language' in df.columns:
        df['language'] = df['language'].astype(str).str.lower().str.strip()
    
    source_df = df[(df['label'] == 0) & (df['language'] == 'python')].copy()
    
    source_df['char_len'] = source_df['code'].str.len()
    source_df = source_df[(source_df['char_len'] > 50) & (source_df['char_len'] < 800)]
    
    if len(source_df) > args.num_samples:
        source_df = source_df.sample(n=args.num_samples, random_state=42)
    
    source_df = source_df.sort_values('char_len').reset_index(drop=True)
    
    logger.info(f"Selected {len(source_df)} samples for OOD Augmentation.")
    
    model, tokenizer = get_model()
    
    # --- TARGET ---
    target_langs = ["c#", "javascript", "go", "c", "php"] 
    
    augmented_rows = []
    
    for i in tqdm(range(0, len(source_df), BATCH_SIZE), desc="Transpiling"):
        batch_df = source_df.iloc[i : i + BATCH_SIZE]
        
        prompts = []
        target_langs_batch = []
        original_codes = []
        original_indices = []
        
        for idx, row in batch_df.iterrows():
            code = row['code']
            t_lang = target_langs[idx % len(target_langs)]
            
            prompt = build_prompt_chat(code, t_lang)
            prompts.append(prompt)
            target_langs_batch.append(t_lang)
            original_codes.append(code)
            original_indices.append(idx)

        if not prompts: continue

        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SOURCE_LEN).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_ids = outputs[:, inputs.input_ids.shape[1]:]
            decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for j, full_output in enumerate(decoded_outputs):
                t_lang = target_langs_batch[j]
                
                virtual_raw = f"```{t_lang}\n{full_output}" 
                
                clean_code = clean_code_block(virtual_raw, t_lang)
                orig_len = len(original_codes[j])
                
                if is_valid_code(clean_code, orig_len):
                    augmented_rows.append({
                        "code": clean_code,
                        "label": 1,
                        "language": t_lang,
                        "aug_method": "transpilation_qwen2.5",
                        "original_source_id": original_indices[j]
                    })
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("OOM Detected. Clearing Cache.")
                torch.cuda.empty_cache()
                continue
            else:
                logger.error(f"Batch Error: {e}")
                continue

    logger.info(f"Augmentation finished. Generated {len(augmented_rows)} OOD samples.")

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        
        # Salviamo
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        aug_df.to_parquet(args.output_path)
        
        logger.info(f"Saved to {args.output_path}")
        logger.info("Distribution of Generated Languages:")
        print(aug_df['language'].value_counts())
        
        # preview
        print("\n--- Preview Data ---")
        print(aug_df[['language', 'label', 'code']].head(3))

if __name__ == "__main__":
    main()