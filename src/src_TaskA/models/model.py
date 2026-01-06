import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from pytorch_metric_learning import losses
from typing import Dict

# =============================================================================
# 1. COMPONENTI MODULARI
# =============================================================================

class AttentionPooler(nn.Module):
    """
    Weighted Attention Pooling.
    Estrae una rappresentazione globale pesata della sequenza.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [Batch, Seq, Hidden]
        
        # Calcolo Score di Attenzione
        x = torch.tanh(self.dense(hidden_states))
        x = self.dropout(x)
        scores = self.out_proj(x).squeeze(-1) # [Batch, Seq]
        
        # Masking: setta i token di padding a -inf
        mask_value = -1e9
        scores = scores.masked_fill(attention_mask == 0, mask_value)
        
        # Softmax per ottenere i pesi (somma = 1)
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1) # [Batch, Seq, 1]
        
        # Somma pesata dei vettori
        pooled_output = torch.sum(hidden_states * attn_weights, dim=1) 
        return pooled_output

class ProjectionHead(nn.Module):
    """
    Proietta l'embedding nello spazio contrastivo (SupCon).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(), 
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # L2 Normalization è fondamentale per la Loss Contrastiva
        return F.normalize(self.net(x), p=2, dim=1)

class StyleProjector(nn.Module):
    """
    Proietta le feature manuali (scalari) in uno spazio vettoriale denso
    da concatenare all'embedding del testo.
    """
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Mish(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.net(x)

class ClassificationHead(nn.Module):
    """
    Head finale.
    """
    def __init__(self, input_dim: int, num_labels: int = 2, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(input_dim, num_labels)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)

# =============================================================================
# 2. MODELLO PRINCIPALE
# =============================================================================

class CodeClassifier(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config_dict = config
        
        model_name = config.get("model_name", "microsoft/unixcoder-base")
        num_labels = config.get("num_labels", 2)
        
        use_extra_features = config.get("use_extra_features", False)
        extra_features_dim = config.get("extra_features_dim", 5)
        style_proj_dim = config.get("style_projection_dim", 64)
        supcon_temp = config.get("supcon_temperature", 0.1)
        
        print(f"[Model] Init UniXcoder ({model_name}) | Hybrid: {use_extra_features}")

        # 1. Text Backbone
        self.encoder = AutoModel.from_pretrained(model_name)
        self.text_hidden_size = self.encoder.config.hidden_size
        
        # 2. Components
        self.pooler = AttentionPooler(self.text_hidden_size)
        self.proj_head = ProjectionHead(self.text_hidden_size) # SupCon lavora sul testo
        
        # 3. Hybrid Strategy
        self.use_extra = use_extra_features
        clf_input_dim = self.text_hidden_size
        
        if self.use_extra:
            self.style_projector = StyleProjector(extra_features_dim, style_proj_dim)
            clf_input_dim += style_proj_dim # Concatenazione: Text + Style
            
        # 4. Classification Head
        self.clf_head = ClassificationHead(clf_input_dim, num_labels=num_labels)
        
        # 5. Losses
        # SupConLoss
        self.supcon_loss_fn = losses.SupConLoss(temperature=supcon_temp)
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.lambda_supcon = config.get("lambda_supcon", 0.5)

        self._init_weights()

    def _init_weights(self):
        """Inizializzazione intelligente per stabilità iniziale."""
        for m in self.clf_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        
        for m in self.proj_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
        if self.use_extra:
            for m in self.style_projector.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input_ids, attention_mask, labels=None, extra_features=None, **kwargs):
        """
        Forward pass ibrido.
        """
        # A. Encoding Testuale
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = self.pooler(outputs.last_hidden_state, attention_mask)
        
        # B. SupCon Feature (Solo Testo)
        # Applichiamo SupCon solo al testo perché vogliamo che l'encoder impari
        # rappresentazioni robuste indipendentemente dalle feature manuali.
        contrastive_features = self.proj_head(text_emb)
        
        # C. Feature Fusion (Per la Classificazione)
        if self.use_extra and extra_features is not None:
            # Proiezione feature stilistiche
            style_emb = self.style_projector(extra_features)
            # Concatenazione [Text Embedding, Style Embedding]
            final_embedding = torch.cat([text_emb, style_emb], dim=1)
        else:
            final_embedding = text_emb
            
        # D. Classificazione
        logits = self.clf_head(final_embedding)
        
        loss = None
        if labels is not None:
            # 1. CE Loss (Sulla rappresentazione finale Ibrida)
            loss_ce = self.ce_loss_fn(logits, labels)
            
            # 2. SupCon Loss (Sulla rappresentazione del Testo)
            # Avvicina i testi umani tra loro, allontana quelli AI
            loss_supcon = self.supcon_loss_fn(contrastive_features, labels)
            
            # 3. Totale
            loss = (1 - self.lambda_supcon) * loss_ce + self.lambda_supcon * loss_supcon
            
        return {
            "loss": loss,
            "logits": logits,
            "embedding": text_emb # Utile solo per visualizzazione
        }