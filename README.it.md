# 🏆 SemEval-2026 Task 13: GenAI Code Detection & Attribution

<div align="center">
  <a href="README.md">
    <img src="https://img.shields.io/badge/Language-English-005BBB?style=for-the-badge&logo=english&logoColor=white" alt="Read in English">
  </a>
</div>

Questo repository contiene la soluzione unificata per la **SemEval-2026 Task 13**, una competizione internazionale incentrata sulla distinzione, attribuzione e analisi del codice sorgente generato da Modelli di Linguaggio (LLM) rispetto a quello scritto da esseri umani.

Il progetto è strutturato in modo modulare per affrontare i **3 Subtask** della competizione, condividendo un ambiente di esecuzione comune e una pipeline di analisi dati centralizzata.

---

## 📌 Panoramica dei Subtask

Il progetto è diviso in tre moduli principali, ognuno con obiettivi e architetture specifiche. Clicca sul nome del task per accedere alla documentazione dettagliata.

| Task | Nome | Obiettivo | Tipo di Problema |
| :--- | :--- | :--- | :--- |
| **[Subtask A](src/src_TaskA/README.md)** | **Machine-Generated Code Detection** | Distinguere se un codice è scritto da un Umano o da una Macchina. | *Binary Classification* |
| **[Subtask B](src/src_TaskB/README.md)** | **Multi-Class Authorship Detection** | Identificare lo specifico modello autore (es. GPT-4, Llama-3). | *Multi-class Classification* |
| **[Subtask C](src/src_TaskC/README.md)** | **Mixed-Source Analysis** | Analizzare modifiche, refactoring e codice ibrido Umano/AI. | *Regression / Hybrid* |

---

## 📂 Struttura del Repository

L'organizzazione delle cartelle è progettata per separare i dati, le immagini di analisi e il codice sorgente.

```bash
.
├── 📁 data/                    # Dataset (parquet) divisi per Task
├── 📁 img/                     # Output visivi degli script di analisi (EDA)
│   ├── 📁 img_TaskA/           # Plot specifici per Task A
│   ├── 📁 img_TaskB/           # Plot specifici per Task B
│   └── 📁 img_TaskC/           # Plot specifici per Task C
│
├── 📁 info_dataset/            # Script per l'analisi statistica dei dati
│   ├── 🐍 info_dataset_subTaskA.py
│   ├── 🐍 info_dataset_subTaskB.py
│   └── 🐍 info_dataset_subtaskC.py
│
├── 📁 src/                     # Codice sorgente dei modelli
│   ├── 📁 src_TaskA/           # Pipeline completa per Subtask A
│   ├── 📁 src_TaskB/           # Pipeline completa per Subtask B
│   └── 📁 src_TaskC/           # Pipeline completa per Subtask C
│
├── 🐍 data.py                  # Scarica il dataset a scelta da kaggle
│
├── 📝 README.md
├── 📄 prepare.sh               # Script di automazione setup (creazione cartelle e env)
├── ⚙️ environment.yml          # Dipendenze Conda condivise
└── ⚙️ .env                     # Variabili d'ambiente (generato da prepare.sh)
```

> [!IMPORTANT]
> Ricordati di generare il `kaggle.json` dal tuo account kaggle:
>
> ```bash
> {"username":"la_tua_username","key":"la_chiave_che_ti_fornisce_kaggle"}
> ```
> 

---

## 🚀 Guida Rapida all'Installazione

Poiché i tre task condividono le stesse dipendenze di base e la stessa struttura, è stato predisposto un setup centralizzato per facilitare l'avvio.

### 1. Prerequisiti
* **Anaconda** o **Miniconda** installati sul sistema.
* Una GPU NVIDIA con driver aggiornati (consigliata per il training).
* Sistema operativo Linux/Mac (o WSL per Windows) per l'esecuzione degli script bash.

### 2. Setup Automatico
Esegui lo script `prepare.sh` dalla root del progetto. Questo script si occuperà di:
1.  Creare la struttura delle directory di output (`results`, `checkpoints`, ecc.).
2.  Generare il file `.env` per le variabili d'ambiente.
3.  Creare e installare l'ambiente virtuale Conda definito in `environment.yml`.

```bash
chmod +x prepare.sh
./prepare.sh
```

### 3. Attivazione dell'Ambiente

Una volta completato il setup, attiva l'ambiente:
```bash
conda activate semeval
```

### 4. Configurazione dei Dati

Apri il file `.env` generato nella root del progetto. Assicurati che la variabile `DATA_PATH` punti alla directory contenente i file `.parquet` (o la cartella scaricata da Kaggle).

Esempio `.env`:
```bash
KAGGLE_USERNAME=Il_tup_username_kaggle
KAGGLE_KEY=la_tua_chiave_kaggle

DATA_PATH=./data
IMG_PATH=./img

COMET_API_KEY=comet_api_key
COMET_PROJECT_NAME=comet_project_name
COMET_WORKSPACE=comet_name_workspace
COMET_EXPERIMENT_NAME=comet_experment_name
```

### 5. Download Dataset

Ricordati di scaricare le dipendenze di kaggle in caso non sia fatto:
```bash
pip install kaggle
```

Scarica il dataset che preferisci da:
```bash
python data.py
```
modifica `competition_name` inserendo il dataset che desideri scaricare da kaggle. In automatico viene scaricato il dataset nella cartella `data`.

---

<div align="center">
  <h2>✨ Autore ✨</h2>

  <p>
    <strong>Giovanni Giuseppe Iacuzzo</strong><br>
    <em>Studente di Ingegneria dell'IA e della Cybersecurity</em><br>
    <em>Università degli Studi Kore di Enna</em>
  </p>

  <p>
    <a href="https://github.com/giovanniIacuzzo" target="_blank">
      <img src="https://img.shields.io/badge/GitHub-GiovanniIacuzzo-181717?style=for-the-badge&logo=github" alt="GitHub"/>
    </a>
    <a href="mailto:giovanni.iacuzzo@unikorestudent.com">
      <img src="https://img.shields.io/badge/Email-Contattami-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/>
    </a>
  </p>
</div>