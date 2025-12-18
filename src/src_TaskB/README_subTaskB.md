# SemEval-2026 Task 13: Subtask B - Multi-Class Authorship Detection

## 📌 Obiettivo del Subtask B

A differenza del Subtask A (binario), il **Subtask B** affronta una sfida di **Fine-Grained Classification**. L'obiettivo è identificare **quale specifico autore** (umano o modello AI) ha generato un determinato snippet di codice.

- **Etichette:** 31+ classi (es. `human`, `gpt-4o`, `llama-3.1-8b`, `starcoder`, ecc.)
- **Input:** Snippet di codice sorgente multilingua
- **Sfida principale:** Il dataset presenta un forte sbilanciamento e include scenari **OOD (Out-Of-Distribution)**.

Il modello deve non solo distinguere tra i generatori noti, ma gestire in modo robusto la presenza di modelli mai visti durante il training nel set di test.

| Setting | Generatori | Obiettivo |
| :--- | :--- | :--- |
| **In-Distribution (ID)** | Presenti nel Train | Classificare correttamente l'autore esatto |
| **Out-Of-Distribution (OOD)** | Assenti nel Train | Generalizzazione o Rilevamento OOD |
| **Class Imbalance** | Human >>> AI Models | Gestire la predominanza della classe `human` |

---

## 📝 Analisi iniziale del dataset

Per analizzare la complessità del task multiclasse, è stato sviluppato lo script `info_dataset_subTaskB.py` che:

1.  Carica i dataset (Train, Validation, Test) e normalizza le etichette dei generatori.
2.  Effettua un controllo sui **duplicati** per evitare *data leakage* tra i set.
3.  Genera metriche avanzate come la **lunghezza in token** (approssimata) e la correlazione **Linguaggio-Generatore**.
4.  Produce il dizionario `GENERATOR_MAP` necessario per la fase di training.

---

### Esempi di risultati salvati in `img_TaskB`:

**1. Distribuzione delle Classi (Train vs Validation)**
Evidenzia il forte sbilanciamento verso la classe *Human* e le differenze di frequenza tra i vari modelli AI.

<div style="text-align:center">
  <img src="../img/img_TaskB/Train_class_dist.png" width="45%" alt="Train Class Distribution"/>
  <img src="../img/img_TaskB/Validation_class_dist.png" width="45%" alt="Validation Class Distribution"/>
</div>

<br>

**2. Verbosità dei Modelli (Lunghezza in Token)**
Confronto della lunghezza media dei codici prodotti. I Boxplot (ordinati per mediana) mostrano come alcuni modelli (es. GPT-4o) tendano a essere più "prolissi" di altri.

<div style="text-align:center">
  <img src="../img/img_TaskB/Train_token_boxplot.png" width="45%" alt="Train Token Boxplot"/>
  <img src="../img/img_TaskB/Validation_token_boxplot.png" width="45%" alt="Validation Token Boxplot"/>
</div>

<br>

**3. Heatmap Normalizzata (Generatore vs Linguaggio)**
Mostra la probabilità condizionata che un certo generatore produca codice in un determinato linguaggio. Utile per identificare modelli specializzati (es. solo Python) rispetto a quelli generalisti.

<div style="text-align:center">
  <img src="../img/img_TaskB/Train_heatmap_norm.png" width="45%" alt="Train Heatmap"/>
  <img src="../img/img_TaskB/Validation_heatmap_norm.png" width="45%" alt="Validation Heatmap"/>
</div>

Queste informazioni aiutano a capire:

- La necessità di tecniche di **re-sampling** o **loss ponderata** dato lo sbilanciamento.
- L'importanza della **lunghezza del codice** come feature discriminante.
- La struttura delle correlazioni tra modelli e linguaggi di programmazione.

---

> [!CAUTION]
> README ANCORA IN FASE DI SVILUPPO...

---

<!--───────────────────────────────────────────────-->
<!--                   AUTORE                     -->
<!--───────────────────────────────────────────────-->

<h2 align="center">✨ Autore ✨</h2>

<p align="center">
  <strong>Giovanni Giuseppe Iacuzzo</strong><br>
  <em>Studente di Ingegneria Dell'IA e della CyberSecurity · Università degli Studi Kore di Enna</em>
</p>

<p align="center">
  <a href="https://github.com/giovanniIacuzzo" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-%40giovanniIacuzzo-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
  <a href="mailto:giovanni.iacuzzo@unikorestudent.com">
    <img src="https://img.shields.io/badge/Email-Contattami-blue?style=for-the-badge&logo=gmail" alt="Email"/>
  </a>
</p>
