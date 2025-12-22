import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
        attn_scores = self.attn(hidden_states).squeeze(-1)
        min_val = -1e4
        attn_scores = attn_scores.masked_fill(attention_mask == 0, min_val)
        attn_weights = F.softmax(attn_scores, dim=-1)
        att_feature = torch.sum(attn_weights.unsqueeze(-1) * hidden_states, dim=1)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_feature = sum_embeddings / sum_mask

        return torch.cat([att_feature, mean_feature], dim=1)

# -----------------------------------------------------------------------------
# 2. Stylometric Network
# -----------------------------------------------------------------------------
class StyloNet(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2), 
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(output_dim * 2, output_dim),
            nn.Mish(),
            nn.Dropout(0.1)
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
        
        self.model_name = model_cfg.get("model_name", "microsoft/unixcoder-base")
        self.num_labels = 2
        
        self.stylo_input_dim = config.get("data", {}).get("stylo_feature_dim", 13)
        
        print(f"Loading backbone: {self.model_name}")
        self.base_model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.hidden_size = self.base_model.config.hidden_size
        
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable()
        
        # --- Configurazione LoRA ---
        self.use_lora = model_cfg.get("use_lora", True) and PEFT_AVAILABLE
        
        if self.use_lora:
            print(f"Activating LoRA for Domain Generalization...")
            # [IMPROVEMENT] Target Modules più completi per RoBERTa/UniXcoder
            # UniXcoder layers: query, key, value, dense, output.dense, etc.
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=32,            # Aumentato R per più capacità
                lora_alpha=64,   # Aumentato Alpha
                lora_dropout=0.1,
                target_modules=["query", "key", "value", "dense", "fc", "out_proj"] 
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
            self.base_model.print_trainable_parameters()
        else:
            print("WARNING: Full Fine-Tuning active. High VRAM usage expected.")

        # --- Components ---
        self.pooler = HybridPooling(self.hidden_size)
        
        self.stylo_hidden_dim = 64
        self.stylo_net = StyloNet(self.stylo_input_dim, self.stylo_hidden_dim)
        
        self.fusion_dim = (self.hidden_size * 2) + self.stylo_hidden_dim
        
        self.supcon_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_size // 2),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, self.num_labels)
        )
        
        self._init_weights(self.classifier)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, stylo_feats=None, labels=None, return_embedding=False):
        # Base Model Forward
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        semantic_emb = self.pooler(outputs.last_hidden_state, attention_mask)
        
        # Stylometry Handling
        if stylo_feats is not None:
            # Assicuriamoci che il device sia corretto
            if stylo_feats.device != semantic_emb.device:
                stylo_feats = stylo_feats.to(semantic_emb.device)
            stylo_emb = self.stylo_net(stylo_feats)
        else:
            device = semantic_emb.device
            dummy_feats = torch.zeros((semantic_emb.size(0), self.stylo_input_dim), device=device)
            stylo_emb = self.stylo_net(dummy_feats)

        # Fusion
        fused_emb = torch.cat([semantic_emb, stylo_emb], dim=1)
        supcon_feats = F.normalize(self.supcon_head(fused_emb), dim=1)

        # Multi-Sample Dropout (Classification Head)
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
            
        if return_embedding:
            return task_logits, loss, supcon_feats, fused_emb
            
        return task_logits, loss

    def compute_metrics(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        
        if preds.ndim > 1: 
            preds = np.argmax(preds, axis=1)
        
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        
        return {
            "accuracy": acc, 
            "precision": p, 
            "recall": r, 
            "f1": f1
        }