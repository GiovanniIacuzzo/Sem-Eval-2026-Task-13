import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# =============================================================================
# 1. COMPONENTI MODULARI
# =============================================================================
class AttentionPooler(nn.Module):
    """
    Sostituisce il pooling standard [CLS]. 
    Calcola una somma pesata degli hidden states basata sulla rilevanza appresa.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        # 1. Calcolo Score di Attenzione per ogni token
        x = torch.tanh(self.dense(hidden_states))
        x = self.dropout(x)
        scores = self.out_proj(x).squeeze(-1) # [Batch, Seq]
        
        # 2. Masking
        mask_value = -1e4
        scores = scores.masked_fill(attention_mask == 0, mask_value)
        
        # 3. Softmax -> Pesi
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1) # [Batch, Seq, 1]
        
        # 4. Somma Pesata
        pooled_output = torch.sum(hidden_states * attn_weights, dim=1) 
        return pooled_output

class FeatureGatingNetwork(nn.Module):
    """
    Processa le features stilometriche (Agnostic Features) e crea un embedding
    che verrà usato per modulare il segnale semantico.
    """
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        return self.net(x)

# =============================================================================
# 2. MODELLO IBRIDO
# =============================================================================
class HybridClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        
        # --- Configurazione Dimensioni ---
        self.num_labels = model_cfg.get("num_labels", 2)
        self.num_handcrafted = data_cfg.get("num_handcrafted_features", 11) 
        model_name = model_cfg.get("base_model", "microsoft/unixcoder-base")
        
        print(f"Initializing Hybrid Model | Backbone: {model_name} | Features: {self.num_handcrafted}")

        # --- 1. Text Backbone ---
        hf_config = AutoConfig.from_pretrained(model_name)
        hf_config.hidden_dropout_prob = 0.2 
        hf_config.attention_probs_dropout_prob = 0.2
        
        self.base_model = AutoModel.from_pretrained(model_name, config=hf_config)
        
        if model_cfg.get("gradient_checkpointing", False):
            self.base_model.gradient_checkpointing_enable()

        self.hidden_size = hf_config.hidden_size

        # --- 2. Modules ---
        self.pooler = AttentionPooler(self.hidden_size)
        
        self.feature_encoder = FeatureGatingNetwork(
            input_dim=self.num_handcrafted, 
            output_dim=128
        )
        
        # --- 3. Fusion Mechanism ---
        fusion_dim = self.hidden_size + 128
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, self.num_labels)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Inizializzazione XAVIER/KAIMING per i layer aggiunti."""
        for m in self.feature_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, extra_features, labels=None):
        # ---------------------------------------------------------------------
        # A. SEMANTIC PATH (Text)
        # ---------------------------------------------------------------------
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        text_embedding = self.pooler(outputs.last_hidden_state, attention_mask)
        
        # ---------------------------------------------------------------------
        # B. AGNOSTIC PATH (Stylometric Features)
        # ---------------------------------------------------------------------
        style_embedding = self.feature_encoder(extra_features) # -> [Batch, 128]
        
        # ---------------------------------------------------------------------
        # C. FUSION & CLASSIFICATION
        # ---------------------------------------------------------------------
        combined_features = torch.cat([text_embedding, style_embedding], dim=1)
        
        logits = self.classifier(combined_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits, labels.view(-1))
            
        return logits, loss, combined_features