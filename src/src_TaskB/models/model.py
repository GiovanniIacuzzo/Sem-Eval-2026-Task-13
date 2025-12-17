import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Gestione import opzionali
try:
    from pytorch_metric_learning import losses
    METRIC_LEARNING_AVAILABLE = True
except ImportError:
    METRIC_LEARNING_AVAILABLE = False
    print("WARNING: pytorch-metric-learning non trovata. Fai 'pip install pytorch-metric-learning'")

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# -----------------------------------------------------------------------------
# DANN Layer (Gradient Reversal)
# -----------------------------------------------------------------------------
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# -----------------------------------------------------------------------------
# Attention Pooling
# -----------------------------------------------------------------------------
class AttentionHead(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attn_scores = self.attn(hidden_states).squeeze(-1)
        # Fix per float16 per evitare NaNs
        min_val = -1e4 if attn_scores.dtype == torch.float16 else -1e9
        attn_scores = attn_scores.masked_fill(attention_mask == 0, min_val)
        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))
        return torch.sum(attn_weights.unsqueeze(-1) * hidden_states, dim=1)

# -----------------------------------------------------------------------------
# CodeClassifier (Updated for Families & Class Weights)
# -----------------------------------------------------------------------------
class CodeClassifier(nn.Module):
    def __init__(self, config, class_weights=None):
        """
        Args:
            config: Dict di configurazione.
            class_weights: Tensor opzionale [11] con i pesi per bilanciare la Loss.
        """
        super().__init__()
        model_cfg = config.get("model", {})
        
        # Base Model
        self.model_name = model_cfg.get("model_name", "microsoft/codebert-base")
        
        # --- CRITICAL CHANGE 1: 11 FAMIGLIE ---
        # 0: Human, 1-10: AI Families
        self.num_labels = 11  
        
        # Mappatura linguaggi per DANN
        self.target_languages = model_cfg.get("languages", ["python", "java", "cpp", "c", "cs", "javascript", "php", "ruby", "rust", "go"]) 
        self.num_languages = len(self.target_languages)
        
        # Backbone Loading
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        base_model = AutoModel.from_pretrained(self.model_name)
        self.hidden_size = base_model.config.hidden_size

        # LoRA Setup
        self.use_lora = model_cfg.get("use_lora", True) and PEFT_AVAILABLE
        if self.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=32,             # R=32 va bene per catturare sfumature complesse
                lora_alpha=64, 
                lora_dropout=0.1, 
                target_modules=["query", "value", "key", "dense"] 
            )
            self.base_model = get_peft_model(base_model, peft_config)
        else:
            self.base_model = base_model

        # --- HEADS ---
        self.pooler = AttentionHead(self.hidden_size)
        
        # 1. Main Classifier (11 Classi)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Mish(),
            nn.Dropout(0.2), 
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        # 2. Projection Head (per SupCon Loss - Generalization)
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 128) 
        )
        
        # 3. Language Classifier (per DANN - Invarianza al linguaggio)
        self.language_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.num_languages)
        )

        # --- CRITICAL CHANGE 2: WEIGHTED LOSS ---
        if class_weights is not None:
            # Se passiamo i pesi (e dobbiamo farlo!), la loss penalizza di più 
            # gli errori sulle classi AI rare rispetto alla classe Human frequente.
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
            
        self.dann_loss = nn.CrossEntropyLoss()
        
        if METRIC_LEARNING_AVAILABLE:
            self.supcon_loss = losses.SupConLoss(temperature=0.1)
        else:
            self.supcon_loss = None

        # Pesi delle loss
        # w_supcon alzato a 0.8 per favorire il clustering delle famiglie (utile per Unseen)
        self.w_supcon = 0.8 
        self.w_dann = 0.5   

    def forward(self, input_ids, attention_mask, lang_ids=None, labels=None, alpha=1.0):
        # Backbone
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        features = self.pooler(outputs.last_hidden_state, attention_mask)

        # Classification Logits
        logits = self.classifier(features)
        
        loss = None
        if labels is not None:
            # 1. Main Loss (Weighted Cross Entropy)
            loss_ce = self.ce_loss(logits, labels)
            
            # 2. Metric Learning Loss (SupCon)
            # Avvicina sample della stessa famiglia, allontana famiglie diverse
            loss_scl = 0.0
            if self.supcon_loss is not None:
                proj = self.projection_head(features)
                proj = F.normalize(proj, p=2, dim=1)
                loss_scl = self.supcon_loss(proj, labels)
            
            # 3. DANN Loss (Adversarial)
            # Rimuove l'informazione del linguaggio di programmazione (Python vs Java)
            # per focalizzarsi sullo stile del generatore.
            loss_dann = 0.0
            if lang_ids is not None and alpha > 0:
                reversed_feats = GradientReversalFn.apply(features, alpha)
                lang_logits = self.language_classifier(reversed_feats)
                
                valid_mask = lang_ids != -1
                if valid_mask.any():
                    loss_dann = self.dann_loss(lang_logits[valid_mask], lang_ids[valid_mask])
            
            # Combine Losses
            loss = loss_ce + (self.w_supcon * loss_scl) + (self.w_dann * loss_dann)
            
        return logits, loss

    def compute_metrics(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        if preds.ndim > 1: preds = np.argmax(preds, axis=1)
        
        acc = accuracy_score(labels, preds)
        # Macro F1: la metrica regina per classi sbilanciate
        f1 = f1_score(labels, preds, average="macro")
        
        # Opzionale: Calcola anche F1 Weighted per avere un'idea della performance globale reale
        f1_weighted = f1_score(labels, preds, average="weighted")
        
        return {"accuracy": acc, "f1_macro": f1, "f1_weighted": f1_weighted}