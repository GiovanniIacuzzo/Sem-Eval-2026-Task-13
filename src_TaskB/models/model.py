import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Tenta di importare PEFT
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not found. Using Full Fine-Tuning (Slower & Heavy).")

# -----------------------------------------------------------------------------
# Advanced Loss Function
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss with optional Label Smoothing.
    Gamma: Focuses on hard examples.
    Smoothing: Prevents overconfidence on specific training examples (helps Unseen task).
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # 1. Label Smoothing Logic
        n_classes = inputs.size(1)
        target_one_hot = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1), 1)
        target_smooth = target_one_hot * (1 - self.label_smoothing) + (self.label_smoothing / n_classes)
        
        # 2. Standard Log Softmax
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        
        # 3. Focal Component
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # 4. Final Loss
        loss = -focal_weight * target_smooth * log_pt
        
        if self.reduction == 'mean':
            return loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum(dim=1)

# -----------------------------------------------------------------------------
# Pooling Layer (The "Brain" of the Classifier)
# -----------------------------------------------------------------------------
class AttentionHead(nn.Module):
    """
    Sostituisce il semplice CLS token.
    Impara a 'pesare' quali token sono importanti nel codice (es. keywords specifiche)
    e crea una rappresentazione globale migliore.
    """
    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [Batch, Seq_Len, Hidden]
        # attention_mask: [Batch, Seq_Len]
        
        # Calcola punteggi di attenzione per ogni token
        attn_scores = self.attn(hidden_states).squeeze(-1) # [Batch, Seq_Len]
        
        # Maschera i token di padding (imposta a -inf)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax per ottenere pesi probabilistici
        attn_weights = F.softmax(attn_scores, dim=-1) # [Batch, Seq_Len]
        attn_weights = self.dropout(attn_weights)
        
        # Somma pesata dei vettori nascosti
        # [Batch, Seq_Len, 1] * [Batch, Seq_Len, Hidden] -> sum -> [Batch, Hidden]
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * hidden_states, dim=1)
        
        return context_vector

# -----------------------------------------------------------------------------
# Main Model Class
# -----------------------------------------------------------------------------
class CodeClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg  = config.get("data", {})
        
        # Base Model Setting (UniXCoder is still King for classification)
        self.model_name = model_cfg.get("model_name", "microsoft/unixcoder-base")
        self.num_labels = 31 
        self.max_length = data_cfg.get("max_length", 512)
        
        # ---------------------------------------------------------------------
        # 1. Load Backbone
        # ---------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base_config = AutoConfig.from_pretrained(self.model_name)
        
        # Load Base Model
        base_model = AutoModel.from_pretrained(self.model_name)
        self.hidden_size = self.base_config.hidden_size

        # ---------------------------------------------------------------------
        # 2. LoRA Setup (Aggressive for T4)
        # ---------------------------------------------------------------------
        self.use_lora = model_cfg.get("use_lora", True) and PEFT_AVAILABLE
        
        if self.use_lora:
            print(f"🚀 Activating Aggressive LoRA for {self.model_name}...")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=64,             # HIGH RANK: Capable of learning fine-grained styles
                lora_alpha=128,   # Alpha = 2*r usually works well
                lora_dropout=0.05,
                # Target ALL linear layers for maximum expressivity
                target_modules=["query", "value", "key", "dense", "fc", "out_proj"] 
            )
            self.base_model = get_peft_model(base_model, peft_config)
            self.base_model.print_trainable_parameters()
        else:
            self.base_model = base_model
            self.base_model.gradient_checkpointing_enable()

        # ---------------------------------------------------------------------
        # 3. Custom Classification Head
        # ---------------------------------------------------------------------
        # Attention Pooling invece di CLS
        self.attention_pooler = AttentionHead(self.hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Mish(), # Funzione di attivazione moderna (meglio di Tanh/Relu)
            nn.Dropout(0.2), # Dropout più alto per evitare overfitting
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        # Inizializzazione pesi della testa
        self.classifier.apply(self._init_weights)

        # ---------------------------------------------------------------------
        # 4. Loss Function
        # ---------------------------------------------------------------------
        # Focal Loss + Label Smoothing
        self.loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass with Attention Pooling.
        """
        # 1. Backbone
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # [Batch, Seq, Hidden]

        # 2. Attention Pooling (Smarter than CLS)
        # Il modello impara quali token guardare
        embedding = self.attention_pooler(last_hidden_state, attention_mask)

        # 3. Classifier
        logits = self.classifier(embedding)
        
        # 4. Loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return logits, loss

    def compute_metrics(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        
        if preds.ndim > 1: 
            preds = np.argmax(preds, axis=1)
        
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        
        return {
            "accuracy": acc, 
            "precision": p, 
            "recall": r, 
            "f1": f1
        }