import pandas as pd
import numpy as np
import re
import math
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

# Configurazione
DATA_DIR = "data/Task_C_Processed"
TRAIN_FILE = "train_processed.parquet"
VAL_FILE = "val_processed.parquet"

def extract_features(code):
    if not isinstance(code, str) or len(code) == 0:
        return [0.0] * 8
        
    features = []
    code_len = len(code) + 1
    
    # 1. Entropia
    counts = Counter(code)
    entropy = -sum((cnt / code_len) * math.log2(cnt / code_len) for cnt in counts.values())
    features.append(entropy / 8.0)
    
    # 2. Special chars
    specials = len(re.findall(r'[{}()\[\];.,+\-*/%&|^!=<>?]', code))
    features.append(specials / code_len)
    
    # 3. Spazi
    features.append(code.count(' ') / code_len)
    
    # 4. Lunghezza parole
    words = re.findall(r'\w+', code)
    num_words = len(words)
    if num_words > 0:
        avg_word_len = sum(len(w) for w in words) / num_words
        unique_ratio = len(set(words)) / num_words
    else:
        avg_word_len = 0
        unique_ratio = 0
    features.append(min(avg_word_len / 20.0, 1.0))

    # 5. Keywords
    keywords = len(re.findall(r'\b(if|for|while|return|def|class|import|void|int|float|string)\b', code))
    features.append(keywords / (num_words + 1))
    
    # 6. Unicità
    features.append(unique_ratio)
    
    # 7. Stringhe lunghe
    long_strings = len(re.findall(r'"[^"]{50,}"|\'[^\']{50,}\'', code))
    features.append(min(long_strings / 5.0, 1.0))
    
    # 8. Nesting Depth (semplificata per velocità)
    current_depth = 0
    max_depth = 0
    for char in code:
        if char == '{': 
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == '}': 
            current_depth = max(0, current_depth - 1)
    features.append(min(max_depth / 10.0, 1.0))
    
    return features

def process_file(filename):
    path = os.path.join(DATA_DIR, filename)
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    
    # Usiamo il parallelismo per velocizzare (usa tutti i core della CPU)
    print(f"Extracting features for {len(df)} samples...")
    codes = df['code'].astype(str).tolist()
    
    # ProcessPoolExecutor per usare tutti i core
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(extract_features, codes), total=len(codes)))
    
    # Aggiungi al dataframe
    df['extra_features'] = results
    
    # Salva
    output_path = os.path.join(DATA_DIR, f"featured_{filename}")
    df.to_parquet(output_path)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    process_file(TRAIN_FILE)
    process_file(VAL_FILE)