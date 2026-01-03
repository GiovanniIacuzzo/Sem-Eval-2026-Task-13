import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('alpha', alpha if alpha is not None else None)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

class CodeClassifier(nn.Module):
    def __init__(self, config, class_weights=None):
        super().__init__()
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        
        self.model_name = model_cfg.get("model_name", "microsoft/unixcoder-base")
        self.num_labels = int(model_cfg.get("num_labels", 10))
        
        hf_config = AutoConfig.from_pretrained(self.model_name)
        hf_config.output_hidden_states = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base_model = AutoModel.from_pretrained(self.model_name, config=hf_config)
        self.hidden_size = hf_config.hidden_size

        print(">> Full Fine-Tuning Enabled (No LoRA)")

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        focal_gamma = float(training_cfg.get("focal_gamma", 2.0))
        print(f"Using Focal Loss (gamma={focal_gamma})")
        self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :] 
        
        logits = self.classifier(cls_embedding)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return logits, loss

    def compute_metrics(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        if preds.ndim > 1: preds = np.argmax(preds, axis=1)
        
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1_macro}