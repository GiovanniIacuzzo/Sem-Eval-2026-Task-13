import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

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
# DANN Layer
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

        if attention_mask is not None:
             attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e4)
        
        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))
        
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * hidden_states, dim=1)
        return context_vector

# -----------------------------------------------------------------------------
# CodeClassifier
# -----------------------------------------------------------------------------
class CodeClassifier(nn.Module):
    def __init__(self, config, class_weights=None):
        """
        Args:
            config: Dict di configurazione nidificato (config['model']...).
            class_weights: Tensor opzionale per bilanciare la Loss (es. per classi rare).
        """
        super().__init__()
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        
        self.model_name = model_cfg.get("model_name", "microsoft/codebert-base")
        self.num_labels = int(model_cfg.get("num_labels", 2))
        
        weights_cfg = training_cfg.get("loss_weights", {})
        self.w_supcon = float(weights_cfg.get("w_supcon", 0.1))
        self.w_dann = float(weights_cfg.get("w_dann", 0.1))
        
        hf_config = AutoConfig.from_pretrained(self.model_name)
        hf_config.output_hidden_states = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        base_model = AutoModel.from_pretrained(self.model_name, config=hf_config)
        self.hidden_size = hf_config.hidden_size

        use_lora = model_cfg.get("use_lora", False)
        if use_lora and PEFT_AVAILABLE:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=int(model_cfg.get("lora_r", 32)),
                lora_alpha=int(model_cfg.get("lora_alpha", 64)),
                lora_dropout=float(model_cfg.get("lora_dropout", 0.1)),
                target_modules=model_cfg.get("target_modules", ["query", "value", "key", "dense"])
            )
            self.base_model = get_peft_model(base_model, peft_config)
            print(f"LoRA Enabled (r={peft_config.r})")
        else:
            self.base_model = base_model
            print("LoRA Disabled or not available.")

        self.pooler = AttentionHead(self.hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Mish(),
            nn.Dropout(0.2), 
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 128) 
        )
        
        num_languages = 20 
        self.language_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, num_languages)
        )

        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
            
        self.dann_loss = nn.CrossEntropyLoss()
        
        if METRIC_LEARNING_AVAILABLE:
            self.supcon_loss = losses.SupConLoss(temperature=0.1)
        else:
            self.supcon_loss = None

    def forward(self, input_ids, attention_mask, lang_ids=None, labels=None, alpha=1.0):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden_state = outputs.last_hidden_state
        features = self.pooler(last_hidden_state, attention_mask)
        logits = self.classifier(features)
        
        loss = None
        if labels is not None:
            loss_ce = self.ce_loss(logits, labels)
            
            loss_scl = 0.0
            if self.supcon_loss is not None and self.w_supcon > 0:
                proj = self.projection_head(features)
                proj = F.normalize(proj, p=2, dim=1)
                loss_scl = self.supcon_loss(proj, labels)
            
            loss_dann = 0.0
            if lang_ids is not None and self.w_dann > 0 and alpha > 0:
                reversed_feats = GradientReversalFn.apply(features, alpha)
                lang_logits = self.language_classifier(reversed_feats)
                
                valid_mask = lang_ids != -1
                if valid_mask.any():
                    loss_dann = self.dann_loss(lang_logits[valid_mask], lang_ids[valid_mask])
            
            loss = loss_ce + (self.w_supcon * loss_scl) + (self.w_dann * loss_dann)
            
        return logits, loss

    def compute_metrics(self, preds, labels):
        """
        Calcola metriche standard (Accuracy, F1 Macro, F1 Weighted).
        """
        preds = np.array(preds)
        labels = np.array(labels)
        
        if preds.ndim > 1: 
            preds = np.argmax(preds, axis=1)
        
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        f1_weighted = f1_score(labels, preds, average="weighted")
        
        return {
            "accuracy": acc, 
            "f1_macro": f1_macro, 
            "f1_weighted": f1_weighted
        }