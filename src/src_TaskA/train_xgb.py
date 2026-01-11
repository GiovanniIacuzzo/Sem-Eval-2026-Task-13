import os
import sys
import argparse
import logging
import json
import numpy as np
import torch
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from typing import Dict, Tuple

# --- CONFIGURAZIONE LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("XGB_Trainer")

STYLE_FEATURE_NAMES = [
    "perplexity_loss", "ttr", "comment_density", "whitespace_ratio",
    "avg_line_len", "std_line_len", "entropy", 
    "id_avg_len", "id_short_ratio", "id_numeric_ratio", "id_char_entropy",
    "case_mix_score", "spacing_inconsistency", "underscore_density",
    "suspicious_comments", "max_line_len"
]

class XGBTrainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.feature_names = None
        
        # Setup device (GPU/CPU) per XGBoost
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tree_method = "hist" 
        logger.info(f"XGBoost configured to use: {self.device.upper()} (method: {self.tree_method})")

    def load_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carica i tensori .pt, esegue la LATE FUSION (concatenazione) 
        e restituisce numpy array pronti per XGBoost.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        logger.info(f"Loading data from {filepath}...")
        data = torch.load(filepath, map_location="cpu")
        
        # 1. Recupera Embeddings [N, 768]
        X_emb = data['embeddings'].numpy()
        
        # 2. Recupera Manual Features [N, 16]
        X_feat = data['features'].numpy()
        
        # 3. LATE FUSION
        X_final = np.hstack([X_emb, X_feat])
        
        # 4. Labels
        y = data['labels'].numpy()
        
        logger.info(f"Loaded {X_final.shape[0]} samples. Feature Dim: {X_final.shape[1]} (768 Emb + {X_feat.shape[1]} Style)")
        return X_final, y

    def train(self, X_train, y_train, X_val, y_val):
        """
        Configura e allena il modello con Early Stopping.
        """
        # Creazione nomi features per l'importanza
        num_emb = X_train.shape[1] - len(STYLE_FEATURE_NAMES)
        emb_names = [f"emb_{i}" for i in range(num_emb)]
        self.feature_names = emb_names + STYLE_FEATURE_NAMES
        
        # Bilanciamento classi
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
        logger.info(f"Class imbalance ratio (Neg/Pos): {ratio:.2f}")

        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'error'],
            'tree_method': self.tree_method,
            'device': self.device,
            'learning_rate': self.args.learning_rate,
            'max_depth': self.args.max_depth,
            'n_estimators': self.args.n_estimators,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': ratio,
            'random_state': 42
        }

        self.model = xgb.XGBClassifier(**params)

        logger.info("Starting Training with Early Stopping...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100
        )
        logger.info("Training finished.")

    def evaluate(self, X_test, y_test, output_dir):
        """
        Valutazione dettagliata sul Test Set.
        """
        logger.info("Evaluating on Test Set...")
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        # 1. Metriche Standard
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        
        logger.info(f"Test Accuracy: {acc:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")
        
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)

        # 2. Salvataggio Metriche su JSON
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "report": report,
            "confusion_matrix": cm.tolist()
        }
        
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # 3. Analisi Feature Importance
        self._save_feature_importance(output_dir)

    def _save_feature_importance(self, output_dir):
        """
        Estrae quali features hanno contato di più.
        Separa e confronta Embeddings vs Features Manuali.
        """
        importance = self.model.feature_importances_
        
        df_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values(by='importance', ascending=False)

        csv_path = os.path.join(output_dir, "feature_importance.csv")
        df_imp.to_csv(csv_path, index=False)
        logger.info(f"Feature importance saved to {csv_path}")

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=df_imp.head(20), palette="viridis")
        plt.title("Top 20 Features for Human vs AI Code Detection")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_features_plot.png"))
        
        df_style = df_imp[df_imp['feature'].isin(STYLE_FEATURE_NAMES)].sort_values(by='importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=df_style, palette="magma")
        plt.title("Impact of Stylo/Forensic Features Only")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "forensics_features_plot.png"))
        
        logger.info("Plots saved.")

    def save_model(self, output_dir):
        path = os.path.join(output_dir, "xgb_model.json")
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors_dir", type=str, required=True, help="Folder containing .pt files")
    parser.add_argument("--output_dir", type=str, default="models/xgb_run")
    parser.add_argument("--n_estimators", type=int, default=1000, help="Max number of trees")
    parser.add_argument("--max_depth", type=int, default=6, help="Tree depth (complex models need more depth)")
    parser.add_argument("--learning_rate", type=float, default=0.05)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    trainer = XGBTrainer(args)
    
    # 1. Caricamento Dati
    train_path = os.path.join(args.vectors_dir, "train_vectors.pt")
    val_path = os.path.join(args.vectors_dir, "val_vectors.pt")
    test_path = os.path.join(args.vectors_dir, "test_vectors.pt")
    
    # Gestione nomi file validation (a volte è validation, a volte val)
    if not os.path.exists(val_path):
        val_path = os.path.join(args.vectors_dir, "validation_vectors.pt")

    X_train, y_train = trainer.load_data(train_path)
    X_val, y_val = trainer.load_data(val_path)
    
    # 2. Training
    trainer.train(X_train, y_train, X_val, y_val)
    
    # 3. Test & Evaluation
    if os.path.exists(test_path):
        X_test, y_test = trainer.load_data(test_path)
        trainer.evaluate(X_test, y_test, args.output_dir)
    else:
        logger.warning("Test set not found. Skipping evaluation.")
    
    # 4. Save
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()