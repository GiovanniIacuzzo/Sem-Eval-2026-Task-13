import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from pytorch_metric_learning import losses
    METRIC_LEARNING_AVAILABLE = True
except ImportError:
    METRIC_LEARNING_AVAILABLE = False

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. Gradient Reversal Layer (DANN)
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
# 2. Focal Loss
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -----------------------------------------------------------------------------
# 3. Attention Pooling
# -----------------------------------------------------------------------------
class AttentionHead(nn.Module):
    """Estrae un vettore riassuntivo pesando l'importanza di ogni token."""
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
        min_val = -1e4 if hidden_states.dtype == torch.float16 else -1e9
        attn_scores = attn_scores.masked_fill(attention_mask == 0, min_val)
        
        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))
        return torch.sum(attn_weights.unsqueeze(-1) * hidden_states, dim=1)

# -----------------------------------------------------------------------------
# 4. Main Model Class
# -----------------------------------------------------------------------------
class CodeClassifier(nn.Module):
    def __init__(self, config, class_weights=None):
        super().__init__()
        model_cfg = config.get("model", {})
        
        self.model_name = model_cfg.get("model_name", "microsoft/graphcodebert-base")
        self.num_labels = 4 
        self.style_feat_dim = 4
        
        print(f"Loading Backbone: {self.model_name}")
        self.transformer_config = AutoConfig.from_pretrained(self.model_name)
        self.base_model = AutoModel.from_pretrained(self.model_name)
        self.hidden_size = self.transformer_config.hidden_size

        self.use_lora = model_cfg.get("use_lora", True) and PEFT_AVAILABLE
        if self.use_lora:
            print("Activating LoRA...")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query", "key", "value", "dense"] 
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
        else:
            self.base_model.gradient_checkpointing_enable()

        self.pooler = AttentionHead(self.hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size + self.style_feat_dim, self.hidden_size),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        self.num_languages = model_cfg.get("num_languages", 12)
        self.language_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.num_languages)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128)
        )

        self._init_weights(self.classifier)
        self._init_weights(self.language_classifier)

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        self.loss_fn = FocalLoss(weight=self.class_weights, gamma=2.0)
        self.dann_loss = nn.CrossEntropyLoss(ignore_index=-1)
        
        if METRIC_LEARNING_AVAILABLE:
            self.supcon_loss = losses.SupConLoss(temperature=0.07)
        else:
            self.supcon_loss = None

        self.contrastive_weight = 0.2
        self.dann_weight = 0.05

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_ids, attention_mask, style_feats=None, lang_ids=None, labels=None, alpha=1.0):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        semantic_features = self.pooler(last_hidden_state, attention_mask)

        if style_feats is not None:
            combined_features = torch.cat((semantic_features, style_feats), dim=1)
        else:
            dummy_style = torch.zeros((semantic_features.size(0), self.style_feat_dim)).to(semantic_features.device)
            combined_features = torch.cat((semantic_features, dummy_style), dim=1)

        task_logits = self.classifier(combined_features)
        
        loss = None
        if labels is not None:
            loss_task = self.loss_fn(task_logits, labels)
            
            loss_scl = 0.0
            if self.supcon_loss is not None:
                proj = F.normalize(self.projection_head(semantic_features), p=2, dim=1)
                loss_scl = self.supcon_loss(proj, labels)
            
            loss_dann = 0.0
            if lang_ids is not None and alpha > 0:
                reversed_feats = GradientReversalFn.apply(semantic_features, alpha)
                lang_logits = self.language_classifier(reversed_feats)
                loss_dann = self.dann_loss(lang_logits, lang_ids)
            
            loss = loss_task + (self.contrastive_weight * loss_scl) + (self.dann_weight * loss_dann)
            
        return task_logits, loss

    def compute_metrics(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        if preds.ndim > 1: preds = np.argmax(preds, axis=1)
        
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        _, _, f1_cls, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        
        metrics = {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
        for i, score in enumerate(f1_cls):
            metrics[f"f1_class_{i}"] = score
        return metrics