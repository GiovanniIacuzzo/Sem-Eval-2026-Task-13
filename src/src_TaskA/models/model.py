import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. Pooling Strategy
# -----------------------------------------------------------------------------
class HybridPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states, attention_mask):
        # Attention Pooling
        attn_scores = self.attn(hidden_states).squeeze(-1)
        min_val = -1e4
        # Maschera i token di padding
        attn_scores = attn_scores.masked_fill(attention_mask == 0, min_val)
        attn_weights = F.softmax(attn_scores, dim=-1)
        att_feature = torch.sum(attn_weights.unsqueeze(-1) * hidden_states, dim=1)

        # Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_feature = sum_embeddings / sum_mask

        # Concatenazione
        return torch.cat([att_feature, mean_feature], dim=1)

# -----------------------------------------------------------------------------
# 2. Stylometric Network
# -----------------------------------------------------------------------------
class StyloNet(nn.Module):
    """
    Rete densa per processare le feature stilometriche numeriche.
    """
    def __init__(self, input_dim, output_dim=128, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# 3. Fusion Model Architecture
# -----------------------------------------------------------------------------
class FusionCodeClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        train_cfg = config.get("training", {})
        
        self.model_name = model_cfg.get("model_name", "microsoft/graphcodebert-base")
        self.num_labels = model_cfg.get("num_labels", 2)
        
        # Stylo Config
        self.stylo_input_dim = data_cfg.get("stylo_feature_dim", 16)
        
        print(f"Loading backbone: {self.model_name}")
        hf_config = AutoConfig.from_pretrained(self.model_name)
        hf_config.output_hidden_states = True
        
        self.base_model = AutoModel.from_pretrained(self.model_name, config=hf_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.hidden_size = self.base_model.config.hidden_size
        
        # Gradient Checkpointing
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable()
        
        self.use_lora = model_cfg.get("use_lora", True) and PEFT_AVAILABLE
        
        if self.use_lora:
            r_dim = model_cfg.get("lora_r", 16)
            alpha = model_cfg.get("lora_alpha", 32)
            dropout = model_cfg.get("lora_dropout", 0.05)
            
            print(f"Activating LoRA: r={r_dim}, alpha={alpha}, dropout={dropout}")
            
            target_modules = ["query", "key", "value", "dense"] 
            
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=r_dim,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
                bias="none"
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
            self.base_model.print_trainable_parameters()
        else:
            print("WARNING: Full Fine-Tuning active. High VRAM usage expected.")

        # --- Components ---
        self.pooler = HybridPooling(self.hidden_size)
        
        self.stylo_hidden_dim = 128
        self.stylo_net = StyloNet(self.stylo_input_dim, self.stylo_hidden_dim)
        
        self.fusion_dim = (self.hidden_size * 2) + self.stylo_hidden_dim
        
        # Multi-Sample Dropout
        clf_dropout = model_cfg.get("extra_dropout", 0.1)
        self.dropouts = nn.ModuleList([nn.Dropout(clf_dropout) for _ in range(5)])
        
        # Classificatore Finale
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Mish(),
            nn.Dropout(clf_dropout),
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        self._init_weights(self.classifier)
        
        ls_val = train_cfg.get("label_smoothing", 0.0)
        print(f"Loss Function initialized with Label Smoothing: {ls_val}")
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=ls_val)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, stylo_feats=None, labels=None):
        # 1. Backbone Forward
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # 2. Pooling Semantico
        semantic_emb = self.pooler(last_hidden, attention_mask)
        
        # 3. Processamento Stilometria
        if stylo_feats is not None:
            if stylo_feats.device != semantic_emb.device:
                stylo_feats = stylo_feats.to(semantic_emb.device)
            stylo_emb = self.stylo_net(stylo_feats)
        else:
            device = semantic_emb.device
            dummy_feats = torch.zeros((semantic_emb.size(0), self.stylo_input_dim), device=device)
            stylo_emb = self.stylo_net(dummy_feats)

        # 4. Fusione
        fused_emb = torch.cat([semantic_emb, stylo_emb], dim=1)

        # 5. Classificazione con Multi-Sample
        task_logits = None
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                task_logits = self.classifier(dropout(fused_emb))
            else:
                task_logits += self.classifier(dropout(fused_emb))
        
        task_logits = task_logits / len(self.dropouts)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(task_logits, labels)
            
        return task_logits, loss