import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
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
    """
    Focal Loss per gestire lo sbilanciamento delle classi e gli esempi difficili.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
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
# 4. Main Model Class
# -----------------------------------------------------------------------------
class CodeClassifier(nn.Module):
    def __init__(self, config, class_weights=None):
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg  = config.get("data", {})
        
        self.model_name = model_cfg.get("model_name", "microsoft/graphcodebert-base")
        self.num_labels = 4 
        
        self.target_languages = model_cfg.get("languages", ["python", "java", "cpp"])
        self.num_languages = len(self.target_languages)
        
        logger_msg = f"Loading Backbone: {self.model_name}"
        print(logger_msg)
        
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.base_model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.hidden_size = self.config.hidden_size

        self.use_lora = model_cfg.get("use_lora", True) and PEFT_AVAILABLE
        
        if self.use_lora:
            print(f"Activating LoRA...")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=32,
                lora_alpha=64,
                lora_dropout=0.1,
                target_modules=["query", "key", "value", "dense"] 
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
            self.base_model.print_trainable_parameters()
        else:
            self.base_model.gradient_checkpointing_enable()

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
        
        self.language_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.num_languages)
        )

        self._init_weights(self.classifier)
        self._init_weights(self.projection_head)
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

        self.contrastive_weight = 0.3
        self.dann_weight = 0.1 

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, lang_ids=None, labels=None, alpha=1.0):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        features = self.pooler(last_hidden_state, attention_mask)

        task_logits = self.classifier(features)
        
        loss = None
        if labels is not None:
            loss_task = self.loss_fn(task_logits, labels)
            
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
            
            loss = loss_task + (self.contrastive_weight * loss_scl) + (self.dann_weight * loss_dann)
            
        return task_logits, loss

    def compute_metrics(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        if preds.ndim > 1: preds = np.argmax(preds, axis=1)
        
        acc = accuracy_score(labels, preds)
        
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        
        _, _, f1_per_class, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        
        metrics = {
            "accuracy": acc, 
            "precision": p, 
            "recall": r, 
            "f1": f1
        }
        
        if len(f1_per_class) >= 4:
            metrics["f1_human"] = f1_per_class[0]
            metrics["f1_ai"] = f1_per_class[1]
            metrics["f1_hybrid"] = f1_per_class[2]
            metrics["f1_adv"] = f1_per_class[3]
            
        return metrics