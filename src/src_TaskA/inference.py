import os
import sys
import torch
import yaml
import argparse
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.src_TaskA.models.model import HybridCodeClassifier
from src.src_TaskA.dataset.dataset import StylometricEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Inference")

class InferencePipeline:
    def __init__(self, model_path: str, config_path: str, device: str = "cuda"):
        self.device = device
        
        # 1. Carica Configurazione
        logger.info(f"Loading config from {config_path}...")
        with open(config_path, "r") as f:
            full_yaml = yaml.safe_load(f)
            self.config = full_yaml.get("config", full_yaml.get("common", full_yaml))

        self.config["semantic_embedding_dim"] = self.config.get("semantic_embedding_dim", 768)
        self.config["structural_feature_dim"] = self.config.get("structural_feature_dim", 10)

        # 2. Inizializza Modello
        logger.info("Initializing Model...")
        self.model = HybridCodeClassifier(self.config)
        
        # 3. Carica Pesi
        logger.info(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        # 4. Componenti per Vettorizzazione (Lazy Loading)
        self.tokenizer = None
        self.encoder = None
        self.style_engine = StylometricEngine()
        
        # Statistiche di normalizzazione (Mean/Std)
        self.train_stats = None

    def _load_encoder(self):
        """Carica UniXcoder solo se necessario per vettorizzare raw text."""
        if self.encoder is None:
            model_name = self.config.get("model_name", "microsoft/unixcoder-base")
            logger.info(f"Loading Backbone {model_name} for inference...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
            self.encoder.eval()

    def load_normalization_stats(self, train_vectors_path: str):
        """
        Carica le statistiche (mean, std) dai vettori di training.
        Fondamentale per la coerenza matematica.
        """
        if os.path.exists(train_vectors_path):
            logger.info(f"Loading normalization stats from {train_vectors_path}...")
            train_data = torch.load(train_vectors_path, map_location="cpu")
            features = train_data["features"].float()
            
            self.train_stats = {
                "mean": features.mean(dim=0).to(self.device),
                "std": (features.std(dim=0) + 1e-6).to(self.device)
            }
        else:
            logger.warning(f"Train vectors not found at {train_vectors_path}. Using Test stats (Less Accurate).")
            self.train_stats = None

    def vectorize_on_the_fly(self, df: pd.DataFrame, batch_size: int = 32):
        """Trasforma il dataframe raw in tensori pronti per il modello."""
        self._load_encoder()
        
        codes = df['code'].astype(str).tolist()
        
        # Gestione Perplexity
        if 'perplexity' not in df.columns:
            logger.warning("'perplexity' column missing in test set! Filling with 0.0.")
            ppls = [0.0] * len(codes)
        else:
            ppls = df['perplexity'].astype(float).tolist()

        # A. Feature Stilometriche
        logger.info("Extracting Structural Features...")
        struct_list = []
        for code, ppl in tqdm(zip(codes, ppls), total=len(codes), desc="Style"):
            struct_list.append(self.style_engine.process(code, ppl))
        
        struct_tensor = torch.tensor(struct_list, dtype=torch.float32).to(self.device)

        # B. Normalizzazione
        if self.train_stats:
            struct_tensor = (struct_tensor - self.train_stats["mean"]) / self.train_stats["std"]
        else:
            # Fallback: Normalizza sul test set stesso
            mean = struct_tensor.mean(dim=0)
            std = struct_tensor.std(dim=0) + 1e-6
            struct_tensor = (struct_tensor - mean) / std

        # C. Embeddings Semantici
        logger.info("Extracting Semantic Embeddings...")
        emb_list = []
        for i in tqdm(range(0, len(codes), batch_size), desc="UniXcoder"):
            batch = codes[i : i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                out = self.encoder(**inputs)
                emb = out.last_hidden_state.mean(dim=1)
            emb_list.append(emb)
        
        emb_tensor = torch.cat(emb_list, dim=0)
        
        return emb_tensor, struct_tensor

    def predict(self, df: pd.DataFrame):
        emb, struct = self.vectorize_on_the_fly(df)
        labels = torch.tensor(df['label'].values, dtype=torch.long).to(self.device)
        
        dataset = TensorDataset(emb, struct, labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        logger.info("Running Inference...")
        with torch.no_grad():
            for b_emb, b_struct, b_lbl in tqdm(loader, desc="Predicting"):
                outputs = self.model(
                    semantic_embedding=b_emb,
                    structural_features=b_struct
                )
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy()) # Probabilità classe AI
                all_labels.extend(b_lbl.cpu().numpy())
                
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def main():
    parser = argparse.ArgumentParser(description="Inference Script for SemEval Task A")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test parquet (e.g., test_sample_ppl_burst.parquet)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model/model_state.bin")
    parser.add_argument("--config_path", type=str, required=True, help="Path to training_meta.yaml or config.yaml")
    parser.add_argument("--train_vectors", type=str, default="data/Task_A/processed/train_vectors.pt", 
                        help="Path to training vectors to retrieve normalization stats")
    parser.add_argument("--output_csv", type=str, default="inference_results.csv", help="Where to save predictions")
    
    args = parser.parse_args()

    # Checks
    if not os.path.exists(args.test_file):
        logger.error(f"Test file not found: {args.test_file}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Init Pipeline
    pipeline = InferencePipeline(args.model_path, args.config_path, device)
    
    # Tenta di caricare statistiche di training
    pipeline.load_normalization_stats(args.train_vectors)
    
    # Load Data
    logger.info(f"Loading Test Data: {args.test_file}")
    df = pd.read_parquet(args.test_file)
    logger.info(f"Test Samples: {len(df)}")
    
    # Esegui Predizione
    preds, labels, probs = pipeline.predict(df)
    
    # Calcolo Metriche
    acc = accuracy_score(labels, preds)
    logger.info(f"\nTest Accuracy: {acc:.4f}")
    
    print("\n" + classification_report(labels, preds, target_names=["Human", "AI"], digits=4))
    
    # Matrice di Confusione
    cm = confusion_matrix(labels, preds)
    print(f"Confusion Matrix:\n{cm}")
    
    # Salvataggio Risultati
    df['predicted_label'] = preds
    df['ai_probability'] = probs
    
    # Aggiungi colonna corretto/errato per analisi errori
    df['is_correct'] = (df['label'] == df['predicted_label'])
    
    logger.info(f"Saving results to {args.output_csv}...")
    df.to_csv(args.output_csv, index=False)
    
    # Analisi Errori Rapida
    errors = df[~df['is_correct']]
    if not errors.empty:
        logger.info(f"Total Errors: {len(errors)}")
        logger.info("Top 3 Error Snippets (High Confidence Wrong):")
        errors['confidence_error'] = np.abs(errors['ai_probability'] - (1 - errors['predicted_label'])) # Distanza da incertezza
        print(errors[['code', 'language', 'label', 'predicted_label', 'ai_probability']].head(3))

if __name__ == "__main__":
    main()