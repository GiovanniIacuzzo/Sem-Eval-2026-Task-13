import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from pytorch_metric_learning import losses
SupConLoss = losses.SupConLoss 

# =============================================================================
# 2. LOSS CUSTOM
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        self.register_buffer('alpha', alpha if alpha is not None else None)

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        # Label Smoothing
        targets_smooth = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_smooth * (1 - self.smoothing) + self.smoothing / inputs.size(-1)
        
        pt = torch.exp(log_probs)
        focal_weight = (1 - pt) ** self.gamma
        
        loss = -focal_weight * targets_smooth * log_probs
        
        if self.alpha is not None:
            if self.alpha.device != loss.device:
                self.alpha = self.alpha.to(loss.device)
            loss = loss * self.alpha.unsqueeze(0)
            
        loss = loss.sum(dim=-1)
        if self.reduction == 'mean': return loss.mean()
        return loss.sum()

# =============================================================================
# 3. COMPONENTI MODULARI (Stile "Modello A")
# =============================================================================

class AttentionPooler(nn.Module):
    """
    Weighted Attention Pooling .
    Il modello impara quali token sono importanti per lo stile,
    invece di usare solo il token [CLS].
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [Batch, Seq, Hidden]
        
        x = torch.tanh(self.dense(hidden_states))
        x = self.dropout(x)
        scores = self.out_proj(x).squeeze(-1) # [Batch, Seq]
        
        # Masking: setta i token di padding a -inf per ignorarli nella softmax
        mask_value = -1e4
        scores = scores.masked_fill(attention_mask == 0, mask_value)
        
        # Softmax per ottenere i pesi (somma = 1)
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1) # [Batch, Seq, 1]
        
        # Somma pesata
        pooled_output = torch.sum(hidden_states * attn_weights, dim=1) 
        return pooled_output

class StyleProjector(nn.Module):
    """
    Proietta le feature manuali (8 feature).
    Usa LayerNorm invece di BatchNorm per stabilità con batch piccoli o misti.
    """
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim), 
            nn.Mish(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.net(x)

class ProjectionHead(nn.Module):
    """
    Head per Contrastive Learning (SupCon).
    Proietta l'embedding semantico in uno spazio ipersferico.
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # L2 Normalize è obbligatorio per SupCon
        return F.normalize(self.net(x), p=2, dim=1)

# =============================================================================
# 4. MODELLO PRINCIPALE
# =============================================================================

class CodeClassifier(nn.Module):
    def __init__(self, config, class_weights=None):
        super().__init__()
        
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        
        self.model_name = model_cfg.get("model_name", "microsoft/unixcoder-base")
        self.num_labels = int(model_cfg.get("num_labels", 11))
        self.num_extra = int(model_cfg.get("num_extra_features", 8))
        self.extra_proj_dim = 64
        
        # --- 1. Backbone ---
        hf_config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base_model = AutoModel.from_pretrained(self.model_name, config=hf_config)
        self.hidden_size = hf_config.hidden_size

        # --- 2. Modules ---
        self.pooler = AttentionPooler(self.hidden_size)
        self.style_projector = StyleProjector(self.num_extra, self.extra_proj_dim)
        self.supcon_head = ProjectionHead(self.hidden_size)

        # --- 3. Classifier Head ---
        combined_dim = self.hidden_size + self.extra_proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        # --- 4. Loss ---
        focal_gamma = float(training_cfg.get("focal_gamma", 2.0))
        smoothing = float(training_cfg.get("label_smoothing", 0.1))
        self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma, smoothing=smoothing)
        
        self._init_weights()

    def _init_weights(self):
        """Inizializza i pesi dei layer aggiunti per una convergenza migliore."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        
        for m in self.style_projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input_ids, attention_mask, labels=None, extra_features=None):
        # A. Encoding Testuale
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Usa Attention Pooling invece di prendere solo il primo token
        semantic_features = self.pooler(outputs.last_hidden_state, attention_mask)

        # B. Style Features
        if extra_features is not None:
            style_features = self.style_projector(extra_features)
        else:
            style_features = torch.zeros(semantic_features.size(0), self.extra_proj_dim).to(semantic_features.device)
        
        # C. Fusion & Classification Logic
        combined_input = torch.cat([semantic_features, style_features], dim=1)
        logits = self.classifier(combined_input)
        
        # D. Feature per SupCon (passata a train.py)
        proj_features = self.supcon_head(semantic_features)
        
        loss = None
        if labels is not None:
            # Calcoliamo qui solo la loss di classificazione
            # La SupCon loss viene calcolata nel loop di training usando proj_features
            loss = self.loss_fn(logits, labels)
            
        return logits, loss, proj_features