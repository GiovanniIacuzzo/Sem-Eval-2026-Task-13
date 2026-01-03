import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class AttentionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        weights = self.attention(last_hidden_state).squeeze(-1)
        weights = weights.masked_fill(attention_mask == 0, float('-inf'))
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)
        return torch.sum(last_hidden_state * weights, dim=1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, smoothing=0.15):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        targets_smooth = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_smooth * (1 - self.smoothing) + self.smoothing / inputs.size(-1)
        ce_loss = (-targets_smooth * log_probs).sum(dim=-1)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        return -(mask * log_prob).sum(1).mean() / (mask.sum(1).mean() + 1e-6)

class CodeClassifier(nn.Module):
    def __init__(self, config, dann_lang_weights=None):
        super().__init__()
        model_cfg = config["model"]
        self.num_languages = model_cfg.get("num_languages", 3)
        self.dann_weight = 0.4
        self.contrastive_weight = 0.6
        
        self.base_model = AutoModel.from_pretrained(model_cfg['model_name'])
        self.hidden_size = self.base_model.config.hidden_size
        self.pooler = AttentionHead(self.hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Mish(), nn.Dropout(0.4), nn.Linear(self.hidden_size // 2, 2)
        )
        self.language_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.BatchNorm1d(self.hidden_size // 2), nn.Mish(),
            nn.Linear(self.hidden_size // 2, self.num_languages)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Mish(), nn.Linear(self.hidden_size // 2, 128)
        )

        self.main_loss_fn = FocalLoss(gamma=2.5, smoothing=0.15)
        self.dann_loss_fn = nn.CrossEntropyLoss(weight=dann_lang_weights)
        self.supcon_loss = SupConLoss()

    def forward(self, input_ids, attention_mask, lang_ids=None, labels=None, alpha=0.0):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        features = self.pooler(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(features)
        
        loss = None
        if labels is not None:
            l_task = self.main_loss_fn(logits, labels)
            proj_feats = F.normalize(self.projection_head(features), p=2, dim=1)
            l_con = self.supcon_loss(proj_feats, labels)
            l_dann = 0.0
            if lang_ids is not None:
                rev_feats = GradientReversalFn.apply(features, alpha)
                l_dann = self.dann_loss_fn(self.language_classifier(rev_feats), lang_ids)
            loss = l_task + (self.contrastive_weight * l_con) + (self.dann_weight * l_dann)
        return loss, logits