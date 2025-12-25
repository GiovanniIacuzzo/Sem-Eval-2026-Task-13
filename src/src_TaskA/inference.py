import os
import sys
import logging
import yaml
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.src_TaskA.models.model import FusionCodeClassifier
from src.src_TaskA.dataset.dataset import load_data
from src.src_TaskA.utils.utils import evaluate

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

# -----------------------------------------------------------------------------
# Main Inference Logic
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{'-'*60}\n{'INFERENCE & TESTING'.center(60)}\n{'-'*60}")

    # 1. Load Configuration
    config_path = "src/src_TaskA/config/config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config non trovato in: {config_path}")
    
    config = load_config(config_path)
    
    # SETUP PERCORSI DATI
    base_dir = "data/Task_A"
    test_path = os.path.join(base_dir, "test_sample.parquet")
    
    if not os.path.exists(test_path):
        logger.warning(f"File di test non trovato in {test_path}. Cerco validation...")
        test_path = config["data"].get("val_path", os.path.join(base_dir, "validation.parquet"))
    
    logger.info(f"Target Test Data: {test_path}")

    config["data"]["val_path"] = test_path

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 3. Initialize Model Structure
    logger.info("Building Model Architecture...")
    model = FusionCodeClassifier(config)

    # 4. Load Best Weights
    checkpoint_dir = config["training"]["checkpoint_dir"]
    weights_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    if os.path.exists(weights_path):
        logger.info(f"Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Nessun checkpoint trovato in {weights_path}. Hai eseguito il train?")

    model.to(device)
    model.eval()

    # 5. Prepare Data Loader
    logger.info("Loading Test Data...")
    _, test_ds, _, _ = load_data(config, model.tokenizer)
    
    test_bs = config["training"]["batch_size"] * 2
    test_dl = DataLoader(
        test_ds, 
        batch_size=test_bs, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    logger.info(f"Test samples loaded: {len(test_ds)}")

    # 6. Run Inference
    logger.info("Running Inference...")
    metrics, true_labels, predictions = evaluate(model, test_dl, device)

    if isinstance(true_labels, list): true_labels = np.array(true_labels)
    elif isinstance(true_labels, torch.Tensor): true_labels = true_labels.cpu().numpy()
    
    if isinstance(predictions, list): predictions = np.array(predictions)
    elif isinstance(predictions, torch.Tensor): predictions = predictions.cpu().numpy()

    # 7. Print Metrics
    print(f"\n{'-'*30}")
    print("TEST RESULTS")
    print(f"{'-'*30}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"{'-'*30}")

    # 8. Detailed Report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, digits=4))

    # 9. Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # DEBUG: Stampiamo alcuni esempi per capire cosa sta sbagliando
    print("\n--- DEBUG: Primi 10 esempi ---")
    print(f"True Labels: {true_labels[:10]}")
    print(f"Predictions: {predictions[:10]}")
    print("------------------------------")

    # Save Confusion Matrix Plot
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(checkpoint_dir, 'confusion_matrix_test.png'))
        logger.info(f"Confusion matrix saved to {checkpoint_dir}")
    except Exception as e:
        logger.warning(f"Impossibile salvare plot matrice: {e}")

    # 10. Save Predictions CSV
    results_df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': predictions,
        'is_correct': true_labels == predictions
    })
    
    output_csv = os.path.join(checkpoint_dir, "test_predictions.csv")
    results_df.to_csv(output_csv, index=False)
    logger.info(f"Predizioni salvate in: {output_csv}")
    
    # 11. Final Accuracy Check
    final_acc = accuracy_score(true_labels, predictions)
    logger.info(f"Accuracy finale calcolata: {final_acc:.4f}")
    logger.info("Inference completed.")