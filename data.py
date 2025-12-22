import os

os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile


# 2. Fix permessi
if os.path.exists("kaggle.json"):
    os.system("chmod 600 kaggle.json")

print("Autenticazione in corso...")

# 3. Autenticazione
try:
    api = KaggleApi()
    api.authenticate()
    print("Autenticazione riuscita!")
except Exception as e:
    print(f"Errore di autenticazione: {e}")
    exit()

# 4. Download
competition_name = "sem-eval-2026-task-13-subtask-a"
target_folder = "data"

print(f"Inizio il download di {competition_name}...")

try:
    # Scarica i file
    api.competition_download_files(competition_name, path=target_folder)
    print("Download completato.")
    
    # 5. Estrazione automatica
    zip_path = os.path.join(target_folder, f"{competition_name}.zip")
    if os.path.exists(zip_path):
        print("Estrazione dello zip in corso...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_folder)
        print(f"File estratti in: {target_folder}")
        os.remove(zip_path) 
    
except Exception as e:
    print(f"Errore durante il download: {e}")
    print("Assicurati di aver accettato le regole della competizione sul sito Kaggle!")