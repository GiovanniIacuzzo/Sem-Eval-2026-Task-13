import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from peft import get_peft_model, LoraConfig, TaskType

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

class CodeModel(nn.Module):
    def __init__(self, config, class_weights=None):
        super().__init__()
        self.config = config
        self.model_name = config.get("model_name", "bigcode/starcoder2-3b")
        self.num_labels = int(config.get("num_labels", 2))
        
        print(f"Loading Base Model: {self.model_name}...")
        hf_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        
        if getattr(hf_config, 'pad_token_id', None) is None:
            hf_config.pad_token_id = hf_config.eos_token_id
        
        self.base_model = AutoModel.from_pretrained(
            self.model_name, 
            config=hf_config, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )
        
        self.base_model.gradient_checkpointing_enable()

        if config.get("use_lora", True):
            print("Applying LoRA...")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False,
                r=config.get("lora_r", 32),
                lora_alpha=config.get("lora_alpha", 64),
                lora_dropout=config.get("lora_dropout", 0.05),
                target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"])
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
            self.base_model.print_trainable_parameters()

        self.hidden_size = hf_config.hidden_size
        self.num_extra = 6
        self.extra_proj_dim = 64

        self.extra_projector = nn.Sequential(
            nn.Linear(self.num_extra, self.extra_proj_dim),
            nn.BatchNorm1d(self.extra_proj_dim),
            nn.Mish(),
            nn.Dropout(0.1)
        )

        combined_dim = self.hidden_size + self.extra_proj_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        # Loss
        self.loss_fn = FocalLoss(alpha=class_weights, gamma=config.get("focal_gamma", 2.0))

    def forward(self, input_ids, attention_mask, labels=None, extra_features=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        last_hidden_state = outputs.last_hidden_state 
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        
        semantic_features = sum_embeddings / sum_mask

        if extra_features is not None:
            extra_features = extra_features.to(dtype=semantic_features.dtype, device=semantic_features.device)
            style_features = self.extra_projector(extra_features)
        else:
            style_features = torch.zeros(
                semantic_features.size(0), self.extra_proj_dim, 
                dtype=semantic_features.dtype, device=semantic_features.device
            )

        combined_input = torch.cat([semantic_features, style_features], dim=1)
        logits = self.classifier(combined_input)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return logits, loss, semantic_features