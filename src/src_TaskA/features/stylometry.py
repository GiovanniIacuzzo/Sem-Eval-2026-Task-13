import re
import math
import numpy as np
from collections import Counter

class StylometryExtractor:
    def __init__(self):
        self.re_camel_case = re.compile(r'\b[a-z]+[A-Z][a-zA-Z0-9]*\b')
        self.re_snake_case = re.compile(r'\b[a-z]+(_[a-z0-9]+)+\b')
        self.re_comments = re.compile(r'(^\s*//|^\s*#|/\*|\s//|\s#)') 
        self.num_features = 13 

    def _calculate_entropy(self, text):
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
        num_non_empty = len(non_empty_lines)
        tokens = [t for t in re.split(r'[^a-zA-Z0-9_]', code_str) if t]
        num_tokens = len(tokens)
        
        # --- FEATURE SET (NORMALIZZATO) ---
        feat_len = np.log1p(code_len) / 10.0
        feat_empty_ratio = (num_lines - num_non_empty) / (num_lines + 1e-5)
        lengths = [len(l) for l in non_empty_lines]
        feat_avg_line_len = np.log1p(np.mean(lengths) if lengths else 0) / 5.0
        feat_entropy = self._calculate_entropy(code_str) / 8.0
        indentations = [len(l) - len(l.lstrip()) for l in non_empty_lines]
        feat_avg_indent = np.log1p(np.mean(indentations) if indentations else 0) / 4.0
        
        count_snake = len([t for t in tokens if self.re_snake_case.match(t)])
        feat_snake_density = count_snake / (num_tokens + 1e-5)
        
        count_camel = len([t for t in tokens if self.re_camel_case.match(t)])
        feat_camel_density = count_camel / (num_tokens + 1e-5)
        
        count_braces = code_str.count('{') + code_str.count('}')
        feat_brace_density = (count_braces / (code_len + 1e-5)) 
        
        count_semicolons = code_str.count(';')
        feat_semicolon_density = (count_semicolons / (code_len + 1e-5)) 
        
        count_comments = len(self.re_comments.findall(code_str))
        feat_comment_density = min(count_comments / (num_lines + 1e-5), 1.0)
        
        count_leading_under = sum(1 for t in tokens if t.startswith('_'))
        feat_leading_under = count_leading_under / (num_tokens + 1e-5)

        alnum_count = sum(c.isalnum() for c in code_str)
        feat_special_ratio = (code_len - alnum_count) / (code_len + 1e-5)
        
        unique_tokens = len(set(tokens))
        feat_ttr = unique_tokens / (num_tokens + 1e-5)

        features = np.array([
            feat_len, feat_empty_ratio, feat_avg_line_len, feat_entropy,
            feat_avg_indent, feat_snake_density, feat_camel_density,
            feat_brace_density, feat_semicolon_density, feat_comment_density,
            feat_leading_under, feat_special_ratio, feat_ttr
        ], dtype=np.float32)
        
        features = np.clip(features, 0.0, 1.0)
        return np.nan_to_num(features)