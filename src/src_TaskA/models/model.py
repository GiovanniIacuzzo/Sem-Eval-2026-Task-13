import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from pytorch_metric_learning import losses
    METRIC_LEARNING_AVAILABLE = True
except ImportError:
    METRIC_LEARNING_AVAILABLE = False
    print("WARNING: pytorch-metric-learning not found. Install with 'pip install pytorch-metric-learning'")

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. Gradient Reversal Layer
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
# 2. Attention Pooling
# -----------------------------------------------------------------------------
class AttentionHead(nn.Module):
    """Pooling intelligente che impara quali token sono importanti."""
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
        
        if attn_scores.dtype == torch.float16:
             min_val = -1e4
        else:
             min_val = -1e9

        attn_scores = attn_scores.masked_fill(attention_mask == 0, min_val)
        
        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))
        return torch.sum(attn_weights.unsqueeze(-1) * hidden_states, dim=1)

# -----------------------------------------------------------------------------
# 3. Model Class
# -----------------------------------------------------------------------------
class CodeClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg  = config.get("data", {})
        
        self.model_name = model_cfg.get("model_name", "microsoft/codebert-base")
        self.num_labels = 2
        self.target_languages = model_cfg.get("languages", ["python", "java", "cpp"])
        self.num_languages = len(self.target_languages)
        self.max_length = data_cfg.get("max_length", 512)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        base_model = AutoModel.from_pretrained(self.model_name)
        self.hidden_size = base_model.config.hidden_size

        self.use_lora = model_cfg.get("use_lora", True) and PEFT_AVAILABLE
        
        if self.use_lora:
            print(f"Activating LoRA for {self.model_name}...")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1,
                target_modules=["query", "value", "key", "dense"]
            )
            self.base_model = get_peft_model(base_model, peft_config)
        else:
            self.base_model = base_model
            self.base_model.gradient_checkpointing_enable()

        # --- Components ---
        self.pooler = AttentionHead(self.hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 128) 
        )
        
        self.language_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.num_languages)
        )

        self._init_weights(self.classifier)
        self._init_weights(self.projection_head)
        self._init_weights(self.language_classifier)

        # --- LOSS FUNCTIONS ---
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.dann_loss = nn.CrossEntropyLoss(ignore_index=-1)
        
        if METRIC_LEARNING_AVAILABLE:
            self.supcon_loss = losses.SupConLoss(temperature=0.07)
        else:
            self.supcon_loss = None

        self.contrastive_weight = 0.5 

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, lang_ids=None, labels=None, alpha=1.0):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        features = self.pooler(outputs.last_hidden_state, attention_mask)

        task_logits = self.classifier(features)
        
        loss = None
        if labels is not None:
            loss_ce = self.ce_loss(task_logits, labels)
            
            loss_scl = 0.0
            if self.supcon_loss is not None:
                proj_features = self.projection_head(features)
                proj_features = F.normalize(proj_features, p=2, dim=1)
                
                loss_scl = self.supcon_loss(proj_features, labels)
            
            loss_dann = 0.0
            if lang_ids is not None and alpha > 0:
                reversed_feats = GradientReversalFn.apply(features, alpha)
                lang_logits = self.language_classifier(reversed_feats)
                loss_dann = self.dann_loss(lang_logits, lang_ids)
            
            loss = loss_ce + (self.contrastive_weight * loss_scl) + loss_dann
            
        return task_logits, loss

    def compute_metrics(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        if preds.ndim > 1: preds = np.argmax(preds, axis=1)
        
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}