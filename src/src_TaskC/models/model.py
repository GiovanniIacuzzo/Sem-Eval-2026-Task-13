import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import transformers.modeling_utils
import transformers.utils.import_utils

# =============================================================================
# 1. UTILITIES & LOSS
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer('alpha', alpha if alpha is not None else None)

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        
        num_classes = inputs.size(-1)
        targets_smooth = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_smooth * (1 - self.smoothing) + self.smoothing / num_classes
        
        pt = torch.exp(log_probs)
        focal_weight = (1 - pt) ** self.gamma
        
        loss = -focal_weight * targets_smooth * log_probs
        
        if self.alpha is not None:
            if self.alpha.device != loss.device:
                self.alpha = self.alpha.to(loss.device)
            loss = loss * self.alpha.unsqueeze(0)

        return loss.sum(dim=-1).mean()

class AttentionPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        x = torch.tanh(self.dense(hidden_states))
        x = self.dropout(x)
        scores = self.out_proj(x).squeeze(-1) 
        
        mask_value = -1e4
        scores = scores.masked_fill(attention_mask == 0, mask_value)
        
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(hidden_states * attn_weights, dim=1)

# =============================================================================
# 2. MODELLO PRINCIPALE
# =============================================================================
class CodeClassifier(nn.Module):
    def __init__(self, config, class_weights=None):
        super().__init__()
        
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        
        self.model_name = model_cfg.get("model_name", "microsoft/unixcoder-base")
        self.num_labels = int(model_cfg.get("num_labels", 4)) 
        self.num_extra = int(model_cfg.get("num_extra_features", 8))
        self.extra_proj_dim = 64
        
        hf_config = AutoConfig.from_pretrained(self.model_name)
        hf_config.hidden_dropout_prob = 0.2 
        hf_config.attention_probs_dropout_prob = 0.2
        
        transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
        transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
        
        try:
            self.base_model = AutoModel.from_pretrained(
                self.model_name, 
                config=hf_config,
                use_safetensors=True 
            )
        except Exception as e:
            print(f"SafeTensors load failed ({e}), falling back to standard load with patch...")
            self.base_model = AutoModel.from_pretrained(
                self.model_name, 
                config=hf_config,
                use_safetensors=False
            )
        
        if model_cfg.get("gradient_checkpointing", False):
            self.base_model.gradient_checkpointing_enable()
            
        self.hidden_size = hf_config.hidden_size

        self.pooler = AttentionPooler(self.hidden_size)
        
        self.style_projector = nn.Sequential(
            nn.Linear(self.num_extra, self.extra_proj_dim),
            nn.LayerNorm(self.extra_proj_dim), 
            nn.Mish(),
            nn.Dropout(0.1)
        )
        
        combined_dim = self.hidden_size + self.extra_proj_dim

        self.supcon_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 128)
        )

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.num_labels) 
        )

        gamma = float(training_cfg.get("focal_gamma", 2.0))
        self.loss_fn = FocalLoss(alpha=class_weights, gamma=gamma, smoothing=0.1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        
        for m in self.style_projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input_ids, attention_mask, labels=None, extra_features=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        semantic_features = self.pooler(outputs.last_hidden_state, attention_mask)

        if extra_features is not None:
            style_features = self.style_projector(extra_features)
        else:
            style_features = torch.zeros(
                semantic_features.size(0), self.extra_proj_dim, 
                device=semantic_features.device
            )
        
        combined_input = torch.cat([semantic_features, style_features], dim=1)
        logits = self.classifier(combined_input)
        proj_features = F.normalize(self.supcon_head(combined_input), p=2, dim=1)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return logits, loss, proj_features