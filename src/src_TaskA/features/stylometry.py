import re
import math
import numpy as np
from collections import Counter

class StylometryExtractor:
    """
    Estrae feature statistiche 'agnostiche' dal codice.
    Ottimizzato per robustezza su linguaggi misti (Python/C++/Go).
    """
    def __init__(self):
        # Regex migliorate
        self.re_camel_case = re.compile(r'\b[a-z]+[A-Z][a-zA-Z0-9]*\b') # Match più preciso per intera parola
        self.re_snake_case = re.compile(r'\b[a-z]+(_[a-z0-9]+)+\b')     # Match più preciso snake_case
        
        # Regex commenti leggermente più robusta
        self.re_comments = re.compile(r'(^\s*//|^\s*#|/\*|\s//|\s#)') 
        
        self.num_features = 13

    def _calculate_entropy(self, text):
        """Calcola l'entropia di Shannon dei caratteri."""
        if not text:
            return 0.0
        counts = Counter(text)
        length = len(text)
        probs = [c / length for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        return entropy

    def extract(self, code_str: str) -> np.ndarray:
        """
        Input: Stringa di codice grezzo.
        Output: Vettore Numpy (float32) di dimensione fissa.
        """
        if not isinstance(code_str, str) or len(code_str.strip()) == 0:
            return np.zeros(self.num_features, dtype=np.float32)

        # Pulizia preliminare
        code_len = len(code_str)
        lines = code_str.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        num_lines = len(lines)
        num_non_empty = len(non_empty_lines)
        
        # Tokenizzazione grezza per analisi lessicale (split su non-alfanumerici)
        tokens = [t for t in re.split(r'[^a-zA-Z0-9_]', code_str) if t]
        num_tokens = len(tokens)
        
        # --- FEATURE SET ---
        
        # 1. Lunghezza Logaritmica
        feat_len = np.log1p(code_len)
        
        # 2. Densità delle righe (Empty Line Ratio)
        feat_empty_ratio = (num_lines - num_non_empty) / (num_lines + 1e-5)
        
        # 3. Lunghezza media delle righe (Log scale)
        lengths = [len(l) for l in non_empty_lines]
        feat_avg_line_len = np.mean(lengths) if lengths else 0
        feat_avg_line_len = np.log1p(feat_avg_line_len)
        
        # 4. Entropia dei caratteri
        feat_entropy = self._calculate_entropy(code_str)
        
        # 5. Indentazione Media
        indentations = [len(l) - len(l.lstrip()) for l in non_empty_lines]
        feat_avg_indent = np.mean(indentations) if indentations else 0
        
        # 6. Snake Case Density (Normalizzato per numero token, non char)
        count_snake = len([t for t in tokens if self.re_snake_case.match(t)])
        feat_snake_density = count_snake / (num_tokens + 1e-5)
        
        # 7. Camel Case Density (Normalizzato per numero token)
        count_camel = len([t for t in tokens if self.re_camel_case.match(t)])
        feat_camel_density = count_camel / (num_tokens + 1e-5)
        
        # 8. Brace Density (Graffe per 100 char)
        count_braces = code_str.count('{') + code_str.count('}')
        feat_brace_density = (count_braces / (code_len + 1e-5)) * 100
        
        # 9. Semicolon Density (Punti e virgola per 100 char)
        count_semicolons = code_str.count(';')
        feat_semicolon_density = (count_semicolons / (code_len + 1e-5)) * 100
        
        # 10. Comment Density (Commenti per riga)
        count_comments = len(self.re_comments.findall(code_str))
        feat_comment_density = count_comments / (num_lines + 1e-5)
        
        # 11. Leading Underscore Ratio (su tokens)
        count_leading_under = sum(1 for t in tokens if t.startswith('_'))
        feat_leading_under = count_leading_under / (num_tokens + 1e-5)

        # 12. Special Character Ratio
        alnum_count = sum(c.isalnum() for c in code_str)
        feat_special_ratio = (code_len - alnum_count) / (code_len + 1e-5)
        
        # 13. Type-Token Ratio (Diversità Lessicale)
        unique_tokens = len(set(tokens))
        feat_ttr = unique_tokens / (num_tokens + 1e-5)

        # --- ASSEMBLAGGIO ---
        features = np.array([
            feat_len,
            feat_empty_ratio,
            feat_avg_line_len,
            feat_entropy,
            feat_avg_indent,
            feat_snake_density,
            feat_camel_density,
            feat_brace_density,
            feat_semicolon_density,
            feat_comment_density,
            feat_leading_under,
            feat_special_ratio,
            feat_ttr
        ], dtype=np.float32)
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

# Test
if __name__ == "__main__":
    extractor = StylometryExtractor()
    sample = "def my_func():\n    # Test comment\n    return 'hello'"
    print(f"Feature Vector Size: {len(extractor.extract(sample))}")
    print(f"Features: {extractor.extract(sample)}")