import os
import sys
import glob
import logging
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging as transformers_logging
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

transformers_logging.set_verbosity_error()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from src.src_TaskA.models.model import CodeClassifier
from src.src_TaskA.dataset.dataset import CodeDataset

class EnsembleInference:
    def __init__(self, config_path, checkpoint_dir, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["model_name"])
        self.models = []
        self.thresholds = []
        self.checkpoint_dir = checkpoint_dir
        
        self._load_ensemble()

    def _load_ensemble(self):
        """Carica tutti i modelli salvati dalla strategia LOLO."""
        model_files = glob.glob(os.path.join(self.checkpoint_dir, "best_lolo_*.pt"))
        
        if not model_files:
            raise FileNotFoundError(f"Nessun modello trovato in {self.checkpoint_dir}")
            
        logger.info(f"Trovati {len(model_files)} modelli per l'Ensemble.")

        num_langs = self.config["model"].get("num_languages", 1)
        dummy_weights = torch.ones(num_langs).to(self.device)

        for path in model_files:
            checkpoint = torch.load(path, map_location=self.device)
            
            model = CodeClassifier(self.config, dann_lang_weights=dummy_weights)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
            self.thresholds.append(checkpoint.get('threshold', 0.5))
            
            lang = checkpoint.get('val_lang', 'unknown')
            logger.info(f"Caricato modello (Val: {lang}) | Thr: {checkpoint.get('threshold', 0.5):.2f}")

        self.avg_threshold = np.mean(self.thresholds)
        logger.info(f"--> Ensemble pronto. Soglia media calcolata: {self.avg_threshold:.4f}")

    def predict(self, test_df):
        """Esegue l'inferenza sul dataframe di test."""
        
        dummy_map = {l: 0 for l in test_df['language'].unique()} if 'language' in test_df else {}
        
        test_ds = CodeDataset(test_df, self.tokenizer, dummy_map, augment=False)
        
        test_dl = DataLoader(
            test_ds, 
            batch_size=self.config["training"]["batch_size"] * 2, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )

        all_probs = []
        true_labels = []
        
        logger.info(f"Inizio inferenza su {len(test_df)} campioni...")
        
        with torch.no_grad():
            for batch in tqdm(test_dl, desc="Inference"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                if "labels" in batch:
                    true_labels.extend(batch["labels"].cpu().numpy())

                batch_probs = []
                
                for model in self.models:
                    logits, _ = model(input_ids, attention_mask, alpha=0.0)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    batch_probs.append(probs)
                
                avg_batch_probs = np.mean(batch_probs, axis=0)
                all_probs.extend(avg_batch_probs)

        return np.array(all_probs), np.array(true_labels)

# -----------------------------------------------------------------------------
# RUNNER PRINCIPALE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    CONFIG_PATH = "src/src_TaskA/config/config.yaml"
    CHECKPOINT_DIR = "./results/results_TaskA/checkpoints" 
    TEST_DATA_PATH = "data/Task_A/test_sample.parquet" 
    
    if os.path.exists(TEST_DATA_PATH):
        logger.info(f"Caricamento dati test da {TEST_DATA_PATH}...")
        try:
            df_test = pd.read_parquet(TEST_DATA_PATH)
            logger.info(f"Dataset caricato: {df_test.shape}")
        except Exception as e:
            logger.error(f"Errore nel caricamento del parquet: {e}")
            sys.exit(1)
    else:
        logger.warning(f"File {TEST_DATA_PATH} non trovato. Genero dati dummy.")
        df_test = pd.DataFrame({
            'text': ["print('hello world')", "def foo(): return 1"] * 50,
            'label': [0, 1] * 50,
            'language': ['python', 'java'] * 50,
            'id': range(100)
        })

    try:
        engine = EnsembleInference(CONFIG_PATH, CHECKPOINT_DIR)
    except Exception as e:
        logger.error(f"Errore inizializzazione Engine: {e}")
        sys.exit(1)
    
    probs, y_true = engine.predict(df_test)
    
    threshold = engine.avg_threshold
    preds = (probs > threshold).astype(int)
    
    print("\n" + "="*60)
    print(" REPORT FINALE DI TEST (ENSEMBLE LOLO)")
    print("="*60)
    
    if len(y_true) > 0 and len(y_true) == len(preds):
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Threshold: {threshold:.4f} (Media dei fold)")
        print("-" * 30)
        print("Classification Report:")
        print(classification_report(y_true, preds, target_names=["Human", "Machine"]))
        
        cm = confusion_matrix(y_true, preds)
        print("Confusion Matrix:")
        print(cm)
        
        if 'language' in df_test.columns:
            print("\n--- Performance per Linguaggio ---")
            df_test['pred'] = preds
            df_test['true'] = y_true
            
            languages = sorted(df_test['language'].unique())
            print(f"{'LANGUAGE':<15} | {'COUNT':<6} | {'F1':<6} | {'ACC':<6}")
            print("-" * 45)
            
            for lang in languages:
                subset = df_test[df_test['language'] == lang]
                if len(subset) > 0:
                    l_f1 = f1_score(subset['true'], subset['pred'])
                    l_acc = accuracy_score(subset['true'], subset['pred'])
                    print(f"{str(lang):<15} | {len(subset):<6} | {l_f1:.4f} | {l_acc:.4f}")

    else:
        print("Nessuna label trovata (o lunghezza mismatch). Salvataggio predizioni raw.")
    
    output_path = "predictions_taskA.csv"
    df_test['probability'] = probs
    df_test['prediction'] = preds
    cols_to_save = ['id', 'language', 'label', 'probability', 'prediction']
    save_df = df_test[[c for c in cols_to_save if c in df_test.columns]]
    save_df.to_csv(output_path, index=False)
    logger.info(f"Predizioni salvate in: {output_path}")