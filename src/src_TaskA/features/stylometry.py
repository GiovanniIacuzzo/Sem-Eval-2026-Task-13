import re
import math
import numpy as np
from collections import Counter

class StylometryExtractor:
    def __init__(self):
        # 1. Regex
        self.re_identifier = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
        
        # Stili di naming specifici
        self.re_camel = re.compile(r'\b[a-z][a-z0-9]*([A-Z][a-z0-9]*)+\b')   # myVariable
        self.re_pascal = re.compile(r'\b[A-Z][a-z0-9]*([A-Z][a-z0-9]*)+\b')  # MyClass
        self.re_snake = re.compile(r'\b[a-z][a-z0-9]*(_[a-z0-9]+)+\b')       # my_variable
        self.re_upper_snake = re.compile(r'\b[A-Z][A-Z0-9]*(_[A-Z0-9]+)+\b') # MY_CONSTANT

        # Tokenizer robusto (esclude solo whitespace)
        self.re_token = re.compile(r'\S+')
        
        # Caratteri speciali (tutto ciò che non è alfanumerico o whitespace)
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
        """Calcola la disuguaglianza (Gini) di un array (es. lunghezze righe)."""
        if not array: return 0.0
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))) if np.sum(array) > 0 else 0.0

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
        
        # Tokenizzazione
        tokens = self.re_token.findall(code_str)
        num_tokens = len(tokens) if tokens else 1
        
        # Identificatori
        identifiers = self.re_identifier.findall(code_str)
        num_identifiers = len(identifiers) if identifiers else 1

        # --- FEATURE EXTRACTION ---

        # Metriche di Volume
        f_len_log = np.log1p(total_chars)
        f_lines_log = np.log1p(num_lines)
        f_tokens_log = np.log1p(num_tokens)

        line_lengths = [len(l) for l in non_empty_lines]
        if line_lengths:
            avg_line_len = np.mean(line_lengths)
            std_line_len = np.std(line_lengths) / (avg_line_len + 1e-5) 
            gini_line_len = self._gini_coefficient(np.array(line_lengths))
        else:
            avg_line_len, std_line_len, gini_line_len = 0.0, 0.0, 0.0
            
        f_avg_line_log = np.log1p(avg_line_len)
        f_cv_line = std_line_len
        f_gini_line = gini_line_len

        indents = [len(l) - len(l.lstrip()) for l in non_empty_lines]
        if indents:
            avg_indent = np.mean(indents)
            std_indent = np.std(indents)
        else:
            avg_indent, std_indent = 0.0, 0.0
            
        f_indent_avg = np.log1p(avg_indent)
        f_indent_std = np.log1p(std_indent)

        # Naming Consistency 
        c_camel = sum(1 for i in identifiers if self.re_camel.match(i))
        c_snake = sum(1 for i in identifiers if self.re_snake.match(i))
        c_pascal = sum(1 for i in identifiers if self.re_pascal.match(i))
        c_upper = sum(1 for i in identifiers if self.re_upper_snake.match(i))
        
        # Qual è lo stile dominante?
        max_style_count = max(c_camel, c_snake, c_pascal, c_upper)
        f_naming_consistency = max_style_count / num_identifiers
        
        # Densità snake e camel
        f_snake_density = c_snake / num_identifiers
        f_camel_density = c_camel / num_identifiers

        num_special = len(self.re_special.findall(code_str))
        f_special_ratio = num_special / total_chars
        
        # Entropia caratteri (Randomness)
        f_entropy = self._calculate_entropy(code_str)
        
        f_ttr = len(set(tokens)) / num_tokens

        f_empty_ratio = (num_lines - num_non_empty) / (num_lines + 1e-5)
        
        # Costruzione Vettore
        features = np.array([
            f_len_log,                   # 0. Lunghezza codice (log)
            f_lines_log,                 # 1. Numero righe (log)
            f_avg_line_log,              # 2. Lunghezza media riga (log)
            f_cv_line,                   # 3. Varianza lunghezza righe (CV)
            f_gini_line,                 # 4. Gini index lunghezza righe
            f_indent_avg,                # 5. Indentazione media
            f_indent_std,                # 6. Varianza indentazione
            f_naming_consistency,        # 7. Coerenza variabili
            f_snake_density,             # 8. Densità Snake case
            f_camel_density,             # 9. Densità Camel case
            f_special_ratio,             # 10. Densità punteggiatura
            f_entropy,                   # 11. Entropia Shannon
            f_ttr,                       # 12. Type-Token Ratio
            f_empty_ratio,               # 13. Ratio righe vuote
            f_tokens_log,                # 14. Num tokens (log)
            num_identifiers / num_tokens # 15. Densità identificatori vs keywords/simboli
        ], dtype=np.float32)

        return np.tanh(features)