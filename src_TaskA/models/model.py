import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -----------------------------------------------------------------------------
# Gradient Reversal Function
# -----------------------------------------------------------------------------
class GradientReversalFn(torch.autograd.Function):
    """
    Gradient Reversal Layer.
    Forward: Identity function (passa i dati senza modificarli).
    Backward: Inverte il gradiente moltiplicandolo per -alpha.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# -----------------------------------------------------------------------------
# Main Model Class
# -----------------------------------------------------------------------------
class CodeClassifier(nn.Module):
    """
    Transformer-based model for Binary Code Classification with DANN & Multi-Sample Dropout.
    
    Architecture:
    1. Backbone: Pre-trained Transformer (UniXCoder/CodeBERT)
    2. Pooling: Mean Pooling
    3. Head A (Task): Multi-Sample Dropout -> MLP -> Human/Machine
    4. Head B (Adversarial): Gradient Reversal -> MLP -> Language Prediction
    """
    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg  = config.get("data", {})
        train_cfg = config.get("training", {})

        self.model_name = model_cfg.get("model_name", "microsoft/codebert-base")
        self.num_labels = model_cfg.get("num_labels", 2)
        
        # Recupera il numero di linguaggi per la testa avversaria
        self.languages = model_cfg.get("languages", ["python", "java", "c++"])
        self.num_languages = len(self.languages)
        
        self.max_length = data_cfg.get("max_length", 256)
        self.device = torch.device(config.get("training_device", "cpu"))

        # Hyperparameters
        self.label_smoothing = train_cfg.get("label_smoothing", 0.0)
        self.freeze_layers   = model_cfg.get("freeze_layers", 0)
        self.extra_dropout   = model_cfg.get("extra_dropout", 0.1)

        # ---------------------------------------------------------------------
        # Backbone Initialization
        # ---------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base_model = AutoModel.from_pretrained(self.model_name)

        # Gradient Checkpointing (Memory optimization)
        # self.base_model.gradient_checkpointing_enable()
        self.hidden_size = self.base_model.config.hidden_size

        # Inject dropout into transformer
        if hasattr(self.base_model.config, "hidden_dropout_prob"):
            self.base_model.config.hidden_dropout_prob = self.extra_dropout
        if hasattr(self.base_model.config, "attention_probs_dropout_prob"):
            self.base_model.config.attention_probs_dropout_prob = self.extra_dropout

        # Layer Freezing
        if hasattr(self.base_model, "roberta") and self.freeze_layers > 0:
            for layer in self.base_model.roberta.encoder.layer[:self.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # ---------------------------------------------------------------------
        # Head 1: Main Task Classifier (Human vs Machine)
        # ---------------------------------------------------------------------
        # Utilizziamo Multi-Sample Dropout (MSD).
        # Creiamo 5 layer di dropout in parallelo.
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.2) for _ in range(5)
        ])
        
        # La parte densa del classificatore (dopo il dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
        # ---------------------------------------------------------------------
        # Head 2: Language Discriminator (Adversarial / DANN)
        # ---------------------------------------------------------------------
        # Questa testa cerca di indovinare il linguaggio. 
        # Noi cercheremo di ingannarla invertendo il gradiente.
        self.language_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_languages)
        )

        # Init weights
        self._init_weights(self.classifier)
        self._init_weights(self.language_classifier)

        self.to(self.device)

    def _init_weights(self, module):
        """Xavier Uniform initialization."""
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

    def forward(self, input_ids, attention_mask, lang_ids=None, labels=None, alpha=1.0):
        """
        Forward pass con DANN e Multi-Sample Dropout.
        
        Args:
            alpha (float): Scaling factor per il Gradient Reversal Layer. 
                           Solitamente parte da 0 e arriva a 1 durante il training.
        """
        # 1. Backbone Pass
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 2. Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask # [batch_size, hidden_size]

        # ---------------------------------------------------------------------
        # 3. Main Task Prediction (Multi-Sample Dropout)
        # ---------------------------------------------------------------------
        # Passiamo l'embedding attraverso 5 dropout diversi e facciamo la media dei logits.
        task_logits_list = []
        for dropout in self.dropouts:
            task_logits_list.append(self.classifier(dropout(mean_pooled)))
        
        # [batch_size, num_labels]
        task_logits = torch.mean(torch.stack(task_logits_list), dim=0)

        # ---------------------------------------------------------------------
        # 4. Adversarial Task Prediction (Language ID)
        # ---------------------------------------------------------------------
        # Applichiamo il Gradient Reversal Layer
        reversed_features = GradientReversalFn.apply(mean_pooled, alpha)
        lang_logits = self.language_classifier(reversed_features)

        # ---------------------------------------------------------------------
        # 5. Loss Calculation
        # ---------------------------------------------------------------------
        loss = None
        if labels is not None:
            # Main Task Loss
            task_loss = self.compute_loss(task_logits, labels)
            
            # Adversarial Loss (se abbiamo le etichette del linguaggio)
            # Nota: durante la validazione/test potremmo non voler calcolare questa loss
            if lang_ids is not None:
                loss_fct = nn.CrossEntropyLoss()
                # La loss avversaria "tira" i pesi per minimizzare l'errore sulla lingua,
                # MA il GRL inverte il gradiente, quindi l'encoder impara a *massimizzare* l'errore (confusione).
                lang_loss = loss_fct(lang_logits, lang_ids)
                
                # Loss Totale
                loss = task_loss + lang_loss
            else:
                loss = task_loss
            
        return task_logits, loss

    def compute_loss(self, logits, labels):
        if self.label_smoothing > 0:
            log_probs = F.log_softmax(logits, dim=-1)
            nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)
            smooth_loss = -log_probs.mean(dim=-1)
            return ((1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss).mean()
        else:
            return nn.CrossEntropyLoss()(logits, labels)

    def compute_metrics(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        
        if preds.ndim > 1: 
            preds = np.argmax(preds, axis=1)
        
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        
        return {
            "accuracy": acc, 
            "precision": p, 
            "recall": r, 
            "f1": f1
        }