import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Tenta di importare PEFT per l'ottimizzazione della memoria (LoRA)
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("PEFT library not found. Falling back to standard fine-tuning (Heavy memory usage).")
    print("Install with: pip install peft")

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-class classification.
    Down-weights well-classified examples and focuses on hard negatives.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CodeClassifier(nn.Module):
    """
    Optimized Transformer Model for Mac M2 (Apple Silicon).
    Uses UniXCoder + LoRA + Focal Loss for efficient 31-class classification.
    """
    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg  = config.get("data", {})
        
        # 1. CHANGE: Default to UniXCoder (State-of-the-Art for Code Understanding)
        self.model_name = model_cfg.get("model_name", "microsoft/unixcoder-base")
        self.num_labels = model_cfg.get("num_labels", 31) # Task B default
        
        self.max_length = data_cfg.get("max_length", 512)
        self.device = torch.device(config.get("training_device", "cpu"))
        
        # Hyperparameters
        self.use_lora = model_cfg.get("use_lora", True) and PEFT_AVAILABLE
        self.extra_dropout = model_cfg.get("extra_dropout", 0.1)

        # ---------------------------------------------------------------------
        # Backbone Initialization
        # ---------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base_config = AutoConfig.from_pretrained(self.model_name)
        
        # Load Base Model
        base_model = AutoModel.from_pretrained(self.model_name)

        # ---------------------------------------------------------------------
        # Memory Optimization Strategy (LoRA vs Full Fine-Tuning)
        # ---------------------------------------------------------------------
        if self.use_lora:
            print(f"🚀 Activating LoRA (Low-Rank Adaptation) for {self.model_name}...")
            # LoRA Configuration
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=8,            # Rank: Higher = more params, better performance. 8 is standard efficient.
                lora_alpha=32,  # Scaling factor
                lora_dropout=0.1,
                # Target modules for UniXCoder (RoBERTa architecture)
                target_modules=["query", "value"] 
            )
            # Wrap the base model
            self.base_model = get_peft_model(base_model, peft_config)
            self.base_model.print_trainable_parameters()
        else:
            # Fallback for standard fine-tuning
            self.base_model = base_model
            # Enable Gradient Checkpointing to save VRAM if not using LoRA
            self.base_model.gradient_checkpointing_enable()

        self.hidden_size = self.base_config.hidden_size

        # ---------------------------------------------------------------------
        # Classification Head
        # ---------------------------------------------------------------------
        # Robust MLP Head
        self.classifier = nn.Sequential(
            nn.Dropout(self.extra_dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.extra_dropout),
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        self._init_weights(self.classifier)

        # ---------------------------------------------------------------------
        # Loss Function
        # ---------------------------------------------------------------------
        # Use Focal Loss to handle the 31-class imbalance
        self.loss_fn = FocalLoss(gamma=2.0)

        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def tokenize(self, texts):
        return self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass using CLS Token Pooling.
        """
        # 1. Backbone Pass
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 2. Pooling Strategy: CLS Token (Index 0)
        # UniXCoder/RoBERTa uses the first token [CLS] (or <s>) as the sentence representation.
        # This is generally superior to Mean Pooling for classification tasks.
        cls_embedding = last_hidden_state[:, 0, :]

        # 3. Classification
        logits = self.classifier(cls_embedding)
        
        # 4. Loss Calculation
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return logits, loss

    def compute_metrics(self, preds, labels):
        """
        Computes accuracy and Macro-F1 (SemEval Metric).
        """
        preds = np.array(preds)
        labels = np.array(labels)
        
        if preds.ndim > 1: 
            preds = np.argmax(preds, axis=1)
        
        acc = accuracy_score(labels, preds)
        
        # 'macro' is crucial for the leaderboard to treat all 31 classes equally
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        
        return {
            "accuracy": acc, 
            "precision": p, 
            "recall": r, 
            "f1": f1
        }