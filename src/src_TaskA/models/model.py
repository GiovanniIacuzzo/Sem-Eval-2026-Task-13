import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class HybridCodeClassifier(nn.Module):
    """
    Lightweight Hybrid Classifier for Pre-Computed Vectors.
    
    Architecture:
    - Input A: Pre-computed Semantic Embeddings (from UniXcoder, dim 768)
    - Input B: Stylometric Features (Manual Engineering, dim 10)
    - Operation: Late Fusion (Concatenation) -> Deep MLP
    
    Note: This model does NOT load the Transformer backbone, ensuring ultra-fast training.
    """
    def __init__(self, config: Dict):
        super().__init__()
        
        # --- Dimensions ---
        # Di default UniXcoder base ha 768, ma rendiamolo configurabile
        self.semantic_dim = config.get("semantic_embedding_dim", 768) 
        self.structural_dim = config.get("structural_feature_dim", 10)
        self.num_labels = config.get("num_labels", 2)
        
        fusion_dim = self.semantic_dim + self.structural_dim
        
        print(f"[Model] Init Hybrid MLP. Semantic: {self.semantic_dim} + Structural: {self.structural_dim} -> Fusion: {fusion_dim}")

        # --- MLP Architecture ---
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.2),
            
            nn.Linear(128, self.num_labels)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Xavier Initialization for better convergence."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, semantic_embedding, structural_features, labels=None, **kwargs):
        """
        Args:
            semantic_embedding: Tensor [Batch, 768] (Già calcolato offline)
            structural_features: Tensor [Batch, 10] (Già calcolato offline)
            labels: Tensor [Batch] (Opzionale)
        """
        
        # 1. Late Fusion
        # Concatenazione diretta dei vettori
        combined_features = torch.cat([semantic_embedding, structural_features], dim=1)
        
        # 2. Classification
        logits = self.classifier(combined_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return {
            "loss": loss,
            "logits": logits
        }