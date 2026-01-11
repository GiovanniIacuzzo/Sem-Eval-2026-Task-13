import os
import logging
import sys
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Configurazioni Globali
REPO_ID = "GiovanniIacuzzo02/SemEval2026_Task13"
DESTINAZIONE = "./checkpoints_scaricati"
FILES_TO_DOWNLOAD = [
    "binary/best_model/model_state.bin",
    "families/best_model/model_state.bin"
]

def get_hf_token() -> str:
    """Carica e valida il token Hugging Face dalle variabili d'ambiente."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    
    if not token:
        logger.error("Token HF_TOKEN non trovato nel file .env o nelle variabili d'ambiente.")
        logger.error("Assicurati di aver creato un file .env con: HF_TOKEN=hf_...")
        sys.exit(1) # Esce con codice di errore
    return token

def download_single_file(token: str, filename: str) -> None:
    """Scarica un singolo file gestendo le eccezioni specifiche."""
    logger.info(f"  Inizio download: {filename}...")
    
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=DESTINAZIONE,
            token=token,
            # force_download=False, # Imposta a True se vuoi sovrascrivere sempre
            # resume_download=True  # Utile per file grandi e connessioni instabili
        )
        logger.info(f"Completato: {local_path}")
        
    except RepositoryNotFoundError:
        logger.error(f"Repository non trovato: {REPO_ID}. Controlla il nome o i permessi.")
    except RevisionNotFoundError:
        logger.error(f"File non trovato nel repo: {filename}")
    except Exception as e:
        logger.error(f"Errore generico durante il download di {filename}: {e}")

def main():
    logger.info("Avvio procedura di download sicuro...")
    
    # 1. Recupero Token
    token = get_hf_token()
    
    # 2. Loop di download sui file configurati
    for filename in FILES_TO_DOWNLOAD:
        download_single_file(token, filename)
        
    logger.info("Procedura terminata.")

if __name__ == "__main__":
    main()