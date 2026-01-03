import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# --- SupCon Loss ---
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return -mean_log_prob_pos.mean()

# --- Gradient Reversal Helper ---
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

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
        min_val = -1e4 if attn_scores.dtype == torch.float16 else -1e9
        attn_scores = attn_scores.masked_fill(attention_mask == 0, min_val)
        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))
        return torch.sum(attn_weights.unsqueeze(-1) * hidden_states, dim=1)

# --- Modello Aggiornato ---
class CodeClassifier(nn.Module):
    def __init__(self, config, dann_lang_weights=None):
        super().__init__()
        model_cfg = config["model"]
        self.num_labels = 2
        self.num_languages = model_cfg.get("num_languages", 1)
        
        # Pesi delle loss
        self.dann_weight = model_cfg.get("dann_weight", 0.2)
        self.contrastive_weight = model_cfg.get("contrastive_weight", 0.5)
        self.use_supcon = model_cfg.get("use_supcon", True)
        
        print(f"Loading Backbone: {model_cfg['model_name']}")
        self.base_model = AutoModel.from_pretrained(model_cfg['model_name'])
        self.hidden_size = self.base_model.config.hidden_size

        if model_cfg.get("use_lora", False):
            lora_cfg = model_cfg["lora"]
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=lora_cfg["r"], 
                lora_alpha=lora_cfg["alpha"], 
                lora_dropout=lora_cfg["dropout"],
                target_modules=lora_cfg.get("target_modules", ["query", "key", "value", "dense"])
            )
            self.base_model = get_peft_model(self.base_model, peft_config)

        self.pooler = AttentionHead(self.hidden_size)
        
        # Task Head principale
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        # Domain Head (DANN)
        self.language_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.num_languages)
        )
        
        # Projection Head per SupCon
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Mish(),
            nn.Linear(self.hidden_size, 128)
        )

        # Losses
        self.main_loss_fn = FocalLoss(gamma=2.0)
        self.dann_loss_fn = nn.CrossEntropyLoss(weight=dann_lang_weights)
        self.supcon_loss = SupConLoss(temperature=model_cfg.get("contrastive_temp", 0.1))

    def forward(self, input_ids, attention_mask, lang_ids=None, labels=None, alpha=0.0):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        features = self.pooler(outputs.last_hidden_state, attention_mask)
        
        logits = self.classifier(features)
        
        loss = None
        if labels is not None:
            # 1. Main Task Loss (Focal)
            loss_task = self.main_loss_fn(logits, labels)
            
            # 2. SupCon Loss (Contrastive)
            loss_con = 0.0
            if self.use_supcon:
                proj_features = F.normalize(self.projection_head(features), p=2, dim=1)
                loss_con = self.supcon_loss(proj_features, labels)
            
            # 3. DANN Loss (Adversarial Language Removal)
            loss_dann = 0.0
            if lang_ids is not None and alpha > 0:
                reversed_feats = GradientReversalFn.apply(features, alpha)
                lang_logits = self.language_classifier(reversed_feats)
                loss_dann = self.dann_loss_fn(lang_logits, lang_ids)
            
            # Loss Totale
            loss = loss_task + (self.contrastive_weight * loss_con) + (self.dann_weight * loss_dann)
            
        return loss, logits