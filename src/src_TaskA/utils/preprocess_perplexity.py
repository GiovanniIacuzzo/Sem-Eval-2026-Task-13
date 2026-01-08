import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import argparse

def calculate_perplexity(model, tokenizer, text, device, max_length=512):
    """
    Calcola la perplexity (esponenziale della CrossEntropyLoss) su uno snippet.
    Perplexity bassa = Il modello lo trova prevedibile (probabile AI).
    Perplexity alta = Il modello è sorpreso (probabile Umano).
    """
    if not text or len(text.strip()) == 0:
        return 0.0
        
    encodings = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
    return torch.exp(loss).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw parquet file (e.g., data/Task_A/train.parquet)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the parquet with perplexity column")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model_name} on {device}...")
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.eval()

    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    
    codes = df['code'].astype(str).tolist()
    perplexities = []
    
    print("Computing Perplexity...")
    for code in tqdm(codes):
        ppl = calculate_perplexity(model, tokenizer, code, device)
        perplexities.append(ppl)
        
    df['perplexity'] = perplexities
    
    print(f"Saving to {args.output_path}...")
    df.to_parquet(args.output_path)
    print("Done.")

if __name__ == "__main__":
    main()