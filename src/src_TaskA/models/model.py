import torch
import torch.nn as nn

class HybridCodeClassifier(nn.Module):
    """
    Balanced Hybrid Classifier.
    
    Architecture:
    - Branch A (Semantic): Raw UniXcoder embedding (768 dim)
    - Branch B (Style): Style Encoder (9 dim -> 128 dim)
    - Fusion: Concatenation -> Deep MLP
    """
    def __init__(self, semantic_dim: int = 768, feature_dim: int = 9, num_labels: int = 2):
        super().__init__()
        
        # --- 1. Style Branch ---
        # Proiettiamo le 9 feature in uno spazio a 128 dimensioni
        self.style_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.Mish()
        )
        
        # --- 2. Fusion Dimensions ---
        # Ora concateniamo 768 (semantica) + 128 (stile upscalato)
        style_out_dim = 128
        fusion_dim = semantic_dim + style_out_dim
        
        print(f"[Model Init] Semantic: {semantic_dim} | Style: {feature_dim}->{style_out_dim} | Fusion Total: {fusion_dim}")

        # --- 3. Main Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_labels)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, semantic_embedding, structural_features, labels=None):
        """
        Args:
            semantic_embedding: [Batch, 768]
            structural_features: [Batch, 9]
        """
        
        # 1. Processiamo le feature stilistiche separatamente
        encoded_style = self.style_encoder(structural_features) # [Batch, 128]
        
        # 2. Concatenazione Bilanciata
        combined_features = torch.cat([semantic_embedding, encoded_style], dim=1)
        
        # 3. Classificazione
        logits = self.classifier(combined_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return {
            "loss": loss,
            "logits": logits
        }