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
    print("Warning: PEFT not found. Running full fine-tuning if LoRA is requested.")

# -----------------------------------------------------------------------------
# 1. Advanced Loss Functions
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss per gestire lo sbilanciamento della difficoltà degli esempi.
    Mette più peso sugli esempi che il modello sbaglia (hard mining implicito).
    """
    def __init__(self, alpha=1, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    Avvicina le rappresentazioni della stessa classe, allontana quelle opposte.
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        features: [batch_size, n_views, features_dim] oppure [batch_size, features_dim]
        labels: [batch_size]
        """
        device = features.device

        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = F.normalize(features, dim=1)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num labels does not match num samples')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(batch_size * features.shape[0] // batch_size).view(-1, 1).to(device), 
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

# -----------------------------------------------------------------------------
# 2. Gradient Reversal (DANN)
# -----------------------------------------------------------------------------
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# -----------------------------------------------------------------------------
# 3. Hybrid Pooling
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
        min_val = -1e4 if hidden_states.dtype == torch.float16 else -1e9
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
# 4. Main Model
# -----------------------------------------------------------------------------
class CodeClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg  = config.get("data", {})
        
        self.model_name = model_cfg.get("model_name", "microsoft/graphcodebert-base")
        self.num_labels = 2
        self.target_languages = model_cfg.get("languages", ["python", "java", "cpp"])
        self.num_languages = len(self.target_languages)
        self.max_length = data_cfg.get("max_length", 512)
        
        print(f"Loading base model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        base_model = AutoModel.from_pretrained(self.model_name)
        self.hidden_size = base_model.config.hidden_size

        self.use_lora = model_cfg.get("use_lora", False) and PEFT_AVAILABLE
        
        if self.use_lora:
            print(f"Activating LoRA (Rank 64)...")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=64,
                lora_alpha=128,
                lora_dropout=0.1,
                target_modules=["query", "value", "key", "dense", "output.dense"] 
            )
            self.base_model = get_peft_model(base_model, peft_config)
            self.base_model.print_trainable_parameters()
        else:
            print("Full Fine-Tuning Mode Activated.")
            self.base_model = base_model
            self.base_model.gradient_checkpointing_enable()

        self.pooler = HybridPooling(self.hidden_size)
        self.pooler_output_size = self.hidden_size * 2 
        
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.pooler_output_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.pooler_output_size, self.hidden_size),
            nn.Mish(),
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        self.language_classifier = nn.Sequential(
            nn.Linear(self.pooler_output_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.num_languages)
        )

        self._init_weights(self.classifier)
        self._init_weights(self.language_classifier)
        self._init_weights(self.projection_head)

        self.focal_loss = FocalLoss(gamma=2.0, label_smoothing=0.05)
        self.supcon_loss = SupConLoss(temperature=0.1) 
        self.dann_loss = nn.CrossEntropyLoss(ignore_index=-1)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, lang_ids=None, labels=None, alpha=1.0, use_supcon=False):
        """
        Args:
            use_supcon (bool): Se True, calcola e aggiunge la Supervised Contrastive Loss.
                               Da attivare DOPO le prime epoche di warmup.
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Feature representation robusta (Hybrid)
        features = self.pooler(outputs.last_hidden_state, attention_mask)

        # Calcolo Logits per classificazione (Multi-Sample Dropout)
        task_logits = None
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                task_logits = self.classifier(dropout(features))
            else:
                task_logits += self.classifier(dropout(features))
        task_logits = task_logits / len(self.dropouts)
        
        loss = None
        if labels is not None:
            # 1. Main Task Loss (Focal Loss)
            loss = self.focal_loss(task_logits, labels)
            
            # 2. Supervised Contrastive Loss (Opzionale/Schedulata)
            if use_supcon:
                # Proiettiamo le feature in uno spazio latente normalizzato
                proj_features = self.projection_head(features)
                # Calcoliamo la loss contrastiva
                # Nota: idealmente per SupCon servono 2 "viste" dello stesso dato nel batch.
                # Qui usiamo un approccio "single-view" supervised che clusterizza le classi,
                # oppure ci affidiamo al Dropout implicito se facciamo 2 forward pass nel training loop.
                loss_scl = self.supcon_loss(proj_features, labels)
                
                # Peso della loss contrastiva (0.1 o 0.5 sono tipici)
                loss = loss + (0.5 * loss_scl)
            
            # 3. Domain Adaptation (DANN) - Opzionale
            if lang_ids is not None and alpha > 0:
                effective_alpha = alpha 
                reversed_feats = GradientReversalFn.apply(features, effective_alpha)
                lang_logits = self.language_classifier(reversed_feats)
                loss_dann = self.dann_loss(lang_logits, lang_ids)
                loss = loss + (0.5 * loss_dann)
            
        return task_logits, loss

    def compute_metrics(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        if preds.ndim > 1: preds = np.argmax(preds, axis=1)
        
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        return {
            "accuracy": acc, 
            "precision": p, 
            "recall": r, 
            "f1": f1
        }