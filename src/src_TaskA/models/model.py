import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from pytorch_metric_learning import losses
from torch.autograd import Function
from typing import Dict

# =============================================================================
# 1. COMPONENTI MODULARI & UTILS
# =============================================================================
class GradientReversal(Function):
    """
    Implementation of the Gradient Reversal Layer (GRL).
    Forward: Identity
    Backward: Multiplies gradient by -alpha
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

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
        x = torch.tanh(self.dense(hidden_states))
        x = self.dropout(x)
        scores = self.out_proj(x).squeeze(-1) # [Batch, Seq]
        
        mask_value = -1e4
        scores = scores.masked_fill(attention_mask == 0, mask_value)
        
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1) # [Batch, Seq, 1]
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
        return F.normalize(self.net(x), p=2, dim=1)

class StyleProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Mish(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.net(x)

class LanguageDiscriminator(nn.Module):
    """
    AVVERSARIO: Cerca di predire il linguaggio di programmazione.
    L'encoder cercherà di ingannare questo modulo.
    """
    def __init__(self, input_dim: int, num_languages: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, num_languages)
        )

    def forward(self, x):
        return self.net(x)

class ClassificationHead(nn.Module):
    """
    TASK: Human vs Machine.
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
        
        # --- Configs ---
        model_name = config.get("model_name", "microsoft/unixcoder-base")
        num_labels = config.get("num_labels", 2)       # Human vs Machine
        num_langs = config.get("num_languages", 3)     # Python, Java, C++ (in train)
        
        use_extra_features = config.get("use_extra_features", False)
        extra_features_dim = config.get("extra_features_dim", 5)
        style_proj_dim = config.get("style_projection_dim", 64)
        
        # Iperparametri Loss
        self.supcon_temp = config.get("supcon_temperature", 0.1)
        self.lambda_supcon = config.get("lambda_supcon", 0.3)
        self.lambda_adv = config.get("lambda_adversarial", 0.1) # Peso dell'avversario
        
        logger_msg = f"[Model] Init UniXcoder ({model_name}) | Hybrid: {use_extra_features} | Adversarial DANN: ON"
        print(logger_msg)

        # 1. Text Backbone
        self.encoder = AutoModel.from_pretrained(model_name)
        self.text_hidden_size = self.encoder.config.hidden_size
        
        # 2. Components
        self.pooler = AttentionPooler(self.text_hidden_size)
        self.proj_head = ProjectionHead(self.text_hidden_size)
        
        # 3. Adversarial Head
        self.lang_discriminator = LanguageDiscriminator(self.text_hidden_size, num_languages=num_langs)
        
        # 4. Feature Fusion
        self.use_extra = use_extra_features
        clf_input_dim = self.text_hidden_size
        
        if self.use_extra:
            self.style_projector = StyleProjector(extra_features_dim, style_proj_dim)
            clf_input_dim += style_proj_dim 
            
        # 5. Main Task Head
        self.clf_head = ClassificationHead(clf_input_dim, num_labels=num_labels)
        
        # 6. Loss Functions
        self.supcon_loss_fn = losses.SupConLoss(temperature=self.supcon_temp)
        self.ce_loss_fn = nn.CrossEntropyLoss()
        
        self._init_weights()

    def _init_weights(self):
        """Inizializzazione pesi custom."""
        for m in self.clf_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        for m in self.lang_discriminator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, input_ids, attention_mask, labels=None, extra_features=None, language_labels=None, **kwargs):
        """
        Forward Pass Complesso:
        1. Encode
        2. SupCon Projection
        3. Adversarial Branch (Language Prediction with Gradient Reversal)
        4. Main Task Branch (Human/Machine Prediction)
        """
        
        # A. Encoding Testuale
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = self.pooler(outputs.last_hidden_state, attention_mask)
        
        # B. SupCon Feature (Projection)
        contrastive_features = self.proj_head(text_emb)
        
        # C. Adversarial Branch (Language Identification)
        loss_adv = torch.tensor(0.0, device=input_ids.device)
        lang_logits = None
        
        if language_labels is not None:
            # APPLICAZIONE GRADIENT REVERSAL
            text_emb_reversed = grad_reverse(text_emb, alpha=1.0) 
            lang_logits = self.lang_discriminator(text_emb_reversed)
            loss_adv = self.ce_loss_fn(lang_logits, language_labels)

        # D. Feature Fusion (Main Task)
        if self.use_extra and extra_features is not None:
            style_emb = self.style_projector(extra_features)
            final_embedding = torch.cat([text_emb, style_emb], dim=1)
        else:
            final_embedding = text_emb
            
        # E. Main Classification
        logits = self.clf_head(final_embedding)
        
        total_loss = None
        losses_log = {}

        if labels is not None:
            # 1. Main Task Loss (Human vs Machine)
            loss_task = self.ce_loss_fn(logits, labels)
            
            # 2. SupCon Loss (Contrastive)
            loss_supcon = self.supcon_loss_fn(contrastive_features, labels)
            
            # 3. Total Loss Combination
            total_loss = loss_task + (self.lambda_supcon * loss_supcon) + (self.lambda_adv * loss_adv)
            
            losses_log = {
                "loss_task": loss_task.item(),
                "loss_supcon": loss_supcon.item(),
                "loss_adv": loss_adv.item()
            }
            
        return {
            "loss": total_loss,
            "logits": logits,
            "detailed_losses": losses_log,
            "embedding": text_emb 
        }