import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        self.register_buffer('alpha', alpha if alpha is not None else None)

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        targets_smooth = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_smooth * (1 - self.smoothing) + self.smoothing / inputs.size(-1)
        
        pt = torch.exp(log_probs)
        focal_weight = (1 - pt) ** self.gamma
        
        loss = -focal_weight * targets_smooth * log_probs
        
        if self.alpha is not None:
            loss = loss * self.alpha.unsqueeze(0)
            
        loss = loss.sum(dim=-1)

        if self.reduction == 'mean': return loss.mean()
        return loss.sum()
    
class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
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
    
class CodeClassifier(nn.Module):
    def __init__(self, config, class_weights=None):
        super().__init__()
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        
        self.model_name = model_cfg.get("model_name", "microsoft/unixcoder-base")
        self.num_labels = int(model_cfg.get("num_labels", 11))
        self.num_extra = 8
        self.extra_proj_dim = 64
        
        hf_config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base_model = AutoModel.from_pretrained(self.model_name, config=hf_config)
        self.hidden_size = hf_config.hidden_size

        self.extra_projector = nn.Sequential(
            nn.Linear(self.num_extra, self.extra_proj_dim),
            nn.BatchNorm1d(self.extra_proj_dim),
            nn.Mish(),
            nn.Dropout(0.1)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Mish(),
            nn.Linear(self.hidden_size // 2, 128)
        )

        combined_dim = self.hidden_size + self.extra_proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        focal_gamma = float(training_cfg.get("focal_gamma", 2.0))
        smoothing = float(training_cfg.get("label_smoothing", 0.1))
        self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma, smoothing=smoothing)

    def forward(self, input_ids, attention_mask, labels=None, extra_features=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        semantic_features = outputs.last_hidden_state[:, 0, :]

        if extra_features is not None:
            style_features = self.extra_projector(extra_features)
        else:
            style_features = torch.zeros(semantic_features.size(0), self.extra_proj_dim).to(semantic_features.device)
        
        combined_input = torch.cat([semantic_features, style_features], dim=1)
        
        logits = self.classifier(combined_input)
        
        proj_features = F.normalize(self.projection_head(semantic_features), p=2, dim=1)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return logits, loss, proj_features