import os
import sys
import logging
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

# -----------------------------------------------------------------------------
# 1. SETUP LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 2. CONFIGURAZIONE UTENTE
# -----------------------------------------------------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "GiovanniIacuzzo02/SemEval2026_Task13" 
TASKS_TO_UPLOAD = [
    {
        "name": "Subtask A (Generalization)",
        "local_path": "results/result_TaskA/checkpoints/best_model", 
        "remote_folder": "SubtaskA_Generalization" 
    },
    {
        "name": "Subtask B (Detection)",
        "local_path": "results/result_TaskB/checkpoints/best_model", 
        "remote_folder": "SubtaskB_Detection"
    }
]

# -----------------------------------------------------------------------------
# 3. FUNZIONI DI UPLOAD
# -----------------------------------------------------------------------------
def check_prerequisites():
    """Verifica che token e cartelle esistano."""
    if not HF_TOKEN:
        logger.error("Token HF_TOKEN non trovato nel file .env o nelle variabili d'ambiente.")
        sys.exit(1)
        
    logger.info(f"Target Repository: https://huggingface.co/{REPO_ID}")

def upload_task(api, task_config):
    """Gestisce l'upload di un singolo task."""
    name = task_config["name"]
    local = task_config["local_path"]
    remote = task_config["remote_folder"]

    logger.info(f"--- Processing: {name} ---")
    
    if not os.path.exists(local):
        logger.warning(f"Cartella locale non trovata: {local}. Salto questo task.")
        return

    logger.info(f"Local: {local}")
    logger.info(f"Remote: {REPO_ID}/{remote}")
    logger.info("Inizio upload (questo può richiedere tempo)...")

    try:
        api.upload_folder(
            folder_path=local,
            repo_id=REPO_ID,
            path_in_repo=remote,
            repo_type="model",
            ignore_patterns=["*.tmp", "__pycache__", "*.git"]
        )
        logger.info(f"Upload completato per {name}!")
        logger.info(f"Link diretto: https://huggingface.co/{REPO_ID}/tree/main/{remote}")
    except Exception as e:
        logger.error(f"Errore critico durante l'upload di {name}: {e}")

def main():
    check_prerequisites()
    
    api = HfApi(token=HF_TOKEN)
    
    try:
        api.create_repo(repo_id=REPO_ID, private=True, exist_ok=True, repo_type="model")
        logger.info("Repository remoto verificato.")
    except Exception as e:
        logger.error(f"Errore accesso/creazione repo: {e}")
        sys.exit(1)

    for task in TASKS_TO_UPLOAD:
        print("-" * 60)
        upload_task(api, task)
    
    print("-" * 60)
    logger.info("Procedura terminata.")

if __name__ == "__main__":
    main()