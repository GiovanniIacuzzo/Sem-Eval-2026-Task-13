import re
import math
import numpy as np
from collections import Counter

class StylometryExtractor:
    def __init__(self):
        # 1. Regex
        self.re_identifier = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
        self.re_camel = re.compile(r'\b[a-z][a-z0-9]*([A-Z][a-z0-9]*)+\b') 
        self.re_pascal = re.compile(r'\b[A-Z][a-z0-9]*([A-Z][a-z0-9]*)+\b') 
        self.re_snake = re.compile(r'\b[a-z][a-z0-9]*(_[a-z0-9]+)+\b')     
        self.re_upper_snake = re.compile(r'\b[A-Z][A-Z0-9]*(_[A-Z0-9]+)+\b') 

        self.re_token = re.compile(r'\S+')
        self.re_special = re.compile(r'[^\w\s]')

        # 16 Feature totali
        self.num_features = 16

    def _calculate_entropy(self, text):
        """Entropia di Shannon sui caratteri."""
        if not text: return 0.0
        counts = Counter(text)
        total = len(text)
        probs = [c / total for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs)

    def _gini_coefficient(self, array):
        if isinstance(array, list):
            array = np.array(array, dtype=np.float32)
            
        if array is None or array.size == 0: 
            return 0.0
            
        array = np.abs(array)
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        
        sum_array = np.sum(array)
        if sum_array == 0: 
            return 0.0
            
        return ((np.sum((2 * index - n - 1) * array)) / (n * sum_array))

    def extract(self, code_str: str) -> np.ndarray:
        # 1. Sanity Check
        if not isinstance(code_str, str) or len(code_str.strip()) < 5:
            return np.zeros(self.num_features, dtype=np.float32)

        # 2. Pre-processing
        code_str = code_str.replace('\r\n', '\n').replace('\r', '\n')
        total_chars = len(code_str)
        lines = code_str.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        num_lines = len(lines)
        num_non_empty = len(non_empty_lines) if non_empty_lines else 1
        
        tokens = self.re_token.findall(code_str)
        num_tokens = len(tokens) if tokens else 1
        
        identifiers = self.re_identifier.findall(code_str)
        num_identifiers = len(identifiers) if identifiers else 1

        # --- FEATURE EXTRACTION ---

        # A. Metriche di Volume
        f_len_log = np.log1p(total_chars)
        f_lines_log = np.log1p(num_lines)
        f_tokens_log = np.log1p(num_tokens)

        # B. Metriche di Layout
        line_lengths = [len(l) for l in non_empty_lines]
        if line_lengths:
            avg_line_len = np.mean(line_lengths)
            std_line_len = np.std(line_lengths)
            # Evitiamo divisione per zero
            f_cv_line = std_line_len / (avg_line_len + 1e-6)
            f_gini_line = self._gini_coefficient(np.array(line_lengths, dtype=np.float32))
        else:
            avg_line_len, f_cv_line, f_gini_line = 0.0, 0.0, 0.0
            
        f_avg_line_log = np.log1p(avg_line_len)

        # C. Indentazione
        indents = [len(l) - len(l.lstrip()) for l in non_empty_lines]
        if indents:
            avg_indent = np.mean(indents)
            std_indent = np.std(indents)
        else:
            avg_indent, std_indent = 0.0, 0.0
            
        f_indent_avg = np.log1p(avg_indent)
        f_indent_std = np.log1p(std_indent)

        # D. Naming Consistency
        c_camel = sum(1 for i in identifiers if self.re_camel.match(i))
        c_snake = sum(1 for i in identifiers if self.re_snake.match(i))
        c_pascal = sum(1 for i in identifiers if self.re_pascal.match(i))
        c_upper = sum(1 for i in identifiers if self.re_upper_snake.match(i))
        
        max_style_count = max(c_camel, c_snake, c_pascal, c_upper)
        f_naming_consistency = max_style_count / (num_identifiers + 1e-6)
        
        f_snake_density = c_snake / (num_identifiers + 1e-6)
        f_camel_density = c_camel / (num_identifiers + 1e-6)

        # E. Complessità
        num_special = len(self.re_special.findall(code_str))
        f_special_ratio = num_special / (total_chars + 1e-6)
        
        f_entropy = self._calculate_entropy(code_str)
        f_ttr = len(set(tokens)) / (num_tokens + 1e-6)

        # F. Empty Lines Ratio
        f_empty_ratio = (num_lines - num_non_empty) / (num_lines + 1e-6)
        
        # 16 Features
        features = np.array([
            f_len_log,                            # 0
            f_lines_log,                          # 1
            f_avg_line_log,                       # 2
            f_cv_line,                            # 3
            f_gini_line,                          # 4
            f_indent_avg,                         # 5
            f_indent_std,                         # 6
            f_naming_consistency,                 # 7
            f_snake_density,                      # 8
            f_camel_density,                      # 9
            f_special_ratio,                      # 10
            f_entropy,                            # 11
            f_ttr,                                # 12
            f_empty_ratio,                        # 13
            f_tokens_log,                         # 14
            num_identifiers / (num_tokens + 1e-6) # 15
        ], dtype=np.float32)

        return np.tanh(features)