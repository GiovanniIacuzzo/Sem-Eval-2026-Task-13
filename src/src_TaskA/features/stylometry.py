import re
import math
import numpy as np
from collections import Counter

class StylometryExtractor:
    def __init__(self):
        # 1. Regex
        self.re_camel_pascal = re.compile(r'\b[a-zA-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b')
        
        # Cattura snake_case
        self.re_snake_case = re.compile(r'\b[a-z]+(_[a-z0-9]+)+\b')
        
        # Cattura costanti
        self.re_const_case = re.compile(r'\b[A-Z]+(_[A-Z0-9]+)+\b')
        
        # Rilevamento commenti (C-style //, /* e Script-style #)
        self.re_comments = re.compile(r'(\/\/|#|\/\*)') 
        
        # Tokenizer (splitta su tutto ciò che non è alfanumerico)
        self.re_tokenizer = re.compile(r'[^a-zA-Z0-9_]+')

        # Numero totale features
        self.num_features = 15 

    def _calculate_entropy(self, text):
        """Calcola l'entropia di Shannon dei caratteri (casualità del testo)."""
        if not text: return 0.0
        counts = Counter(text)
        length = len(text)
        probs = [c / length for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs)

    def extract(self, code_str: str) -> np.ndarray:
        if not isinstance(code_str, str) or len(code_str.strip()) == 0:
            return np.zeros(self.num_features, dtype=np.float32)

        code_len = len(code_str)
        lines = code_str.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        num_lines = len(lines)
        num_non_empty = len(non_empty_lines) if non_empty_lines else 1
        
        tokens = [t for t in self.re_tokenizer.split(code_str) if t.strip()]
        num_tokens = len(tokens) if tokens else 1
        
        # --- CALCOLO FEATURES ---
        feat_len = np.log1p(code_len) / 12.0
        
        feat_empty_ratio = (num_lines - num_non_empty) / (num_lines + 1e-5)
        
        lengths = [len(l) for l in non_empty_lines]
        if lengths:
            avg_len = np.mean(lengths)
            std_len = np.std(lengths)
        else:
            avg_len, std_len = 0, 0
            
        feat_avg_line_len = np.log1p(avg_len) / 6.0 
        feat_std_line_len = np.log1p(std_len) / 4.0

        # Entropia
        feat_entropy = self._calculate_entropy(code_str) / 8.0
        
        # Analisi Indentazione
        indentations = [len(l) - len(l.lstrip()) for l in non_empty_lines]
        if indentations:
            avg_indent = np.mean(indentations)
            max_indent = np.max(indentations)
        else:
            avg_indent, max_indent = 0, 0
            
        feat_avg_indent = np.log1p(avg_indent) / 4.0
        feat_max_indent = np.log1p(max_indent) / 5.0
        
        count_snake = sum(1 for t in tokens if self.re_snake_case.match(t))
        count_camel = sum(1 for t in tokens if self.re_camel_pascal.match(t))
        count_const = sum(1 for t in tokens if self.re_const_case.match(t))
        
        # Snake Case Density
        feat_snake_density = count_snake / num_tokens
        feat_camel_density = count_camel / num_tokens
        feat_const_density = count_const / num_tokens
        
        # Analisi Sintattica
        count_braces = code_str.count('{') + code_str.count('}')
        count_semicolons = code_str.count(';')
        
        # Brace Densitys
        feat_brace_density = np.log1p(count_braces) / (np.log1p(code_len) + 1e-5)
        
        # Semicolon Density
        feat_semicolon_density = np.log1p(count_semicolons) / (np.log1p(code_len) + 1e-5)
        
        # Comment Density
        count_comments = len(self.re_comments.findall(code_str))
        feat_comment_density = min(count_comments / (num_lines + 1e-5), 1.0)
        
        # Special Character Ratio
        alnum_count = sum(c.isalnum() for c in code_str)
        feat_special_ratio = (code_len - alnum_count) / (code_len + 1e-5)
        
        # TTR (Type-Token Ratio)
        unique_tokens = len(set(tokens))
        feat_ttr = unique_tokens / num_tokens

        # Assemblaggio Features
        features = np.array([
            feat_len,             
            feat_empty_ratio,     
            feat_avg_line_len,    
            feat_std_line_len,
            feat_entropy,         
            feat_avg_indent,      
            feat_max_indent, 
            feat_snake_density,   
            feat_camel_density,   
            feat_const_density,
            feat_brace_density,    
            feat_semicolon_density,
            feat_comment_density,  
            feat_special_ratio,    
            feat_ttr               
        ], dtype=np.float32)
        

        features = np.tanh(features * 2.0)
        
        return np.nan_to_num(features)