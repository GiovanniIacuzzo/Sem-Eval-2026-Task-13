import torch
import logging
import sys
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# 1. SETUP & LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # A. Identifica colonna TESTO
        if "text" in df.columns: self.text_col = "text"
        elif "content" in df.columns: self.text_col = "content"
        elif "code" in df.columns: self.text_col = "code"
        else:
            raise ValueError(f"Columns error. Found: {list(df.columns)}")
        
        # B. Identifica colonna LABEL (Ground Truth)
        self.label_source = None
        
        if "family" in df.columns:
            self.label_source = "family"
            self.use_mapping_func = False
        elif "generator" in df.columns:
            self.label_source = "generator"
            self.use_mapping_func = True
        elif "label" in df.columns:
            self.label_source = "label"
            self.use_mapping_func = False
            logger.warning("Using numeric 'label' column. Metrics might be wrong if not mapped to strings!")
            
        logger.info(f"Text Column: '{self.text_col}' | Label Source: '{self.label_source}'")
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        
        # Tokenizzazione
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        extra_feats = torch.zeros(8)
        
        final_label = "Unknown"
        if self.label_source:
            raw_val = row[self.label_source]
            
            if self.use_mapping_func:
                final_label = get_family_name(raw_val)
            else:
                final_label = str(raw_val)
                if final_label.lower() == 'human':
                    final_label = "Human"

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "extra_features": extra_feats,
            "original_label": final_label
        }

def get_family_name(generator_str):
    """
    Converte la stringa raw del generatore (es. 'gpt-4') nel nome della famiglia (es. 'gpt').
    Copiato dal preprocessing originale per coerenza.
    """
    gen = str(generator_str).lower().strip()
    
    # 0. Human Check
    if 'human' in gen: return 'Human' 
    
    # 1. Family Checks
    if 'granite' in gen or 'ibm' in gen: return 'granite'
    if 'llama' in gen: return 'llama'
    if 'gpt' in gen or 'openai' in gen: return 'gpt'
    if 'mistral' in gen or 'codestral' in gen: return 'mistral'
    if 'qwen' in gen: return 'qwen'
    if 'phi' in gen: return 'phi'
    if 'deepseek' in gen: return 'deepseek'
    if 'gemma' in gen: return 'gemma'
    if 'yi' in gen and 'yi-' in gen: return 'yi'
    if 'starcoder' in gen or 'bigcode' in gen or 'santa' in gen: return 'starcoder'
    
    return 'other'