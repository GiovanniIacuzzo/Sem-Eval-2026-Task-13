# SemEval-2026 Task 13: Subtask C - Mixed-Source & Code Modification Analysis

## 📌 Subtask C Objective

<div align="center">
  <a href="README.it.md">
    <img src="https://img.shields.io/badge/Lingua-Italiano-008C45?style=for-the-badge&logo=italian&logoColor=white" alt="Leggi in Italiano">
  </a>
</div>

**Subtask C** introduces the highest level of complexity: the analysis of **hybrid** or **modified** code. We no longer ask just "who wrote it," but analyze the nuances of collaboration between Human and Machine (e.g., human code refactored by AI, or AI code fixed by humans).

- **Input:** Code snippets (with potential differences/diffs or multiple versions).
- **Target:** Hybrid Classification or Regression (e.g., identifying if code was modified by an AI or assigning an "artificiality" score).
- **Key Challenge:** Detecting subtle modification patterns that do not alter the code logic but change its style (Refactoring, Obfuscation, Translation).

| Setting | Analysis Type | Objective |
| :--- | :--- | :--- |
| **Mixed Sources** | Human + AI | Detect boundaries or AI contribution percentages |
| **Refactoring** | Original vs Modified | Understand if the style was altered by a model |
| **Soft-Labeling** | Score 0.0 - 1.0 | Assign a degree of certainty regarding the origin |

---

## 📝 Initial Dataset Analysis

To address the heterogeneous nature of this task, the `info_dataset_subTaskC.py` script was designed to dynamically adapt to the target type (categorical or numerical):

1.  Automatically detects if the target is a **Class** (Bar charts) or a **Score** (Histograms/KDE).
2.  Analyzes code length while handling outliers (95th percentile cut-off).
3.  Examines language distribution to understand if the modification task is language-specific or agnostic.

---

### Examples of results saved in `img_TaskC`:

**1. Target Distribution (Label or Score)**
Unlike previous tasks, here we might observe continuous distributions (regression scores) or hybrid classes. This chart is crucial for choosing the Loss Function (CrossEntropy vs MSE).

<div style="text-align:center">
  <img src="../../img/img_TaskC/Train_target_dist.png" width="45%" alt="Train Target Distribution"/>
  <img src="../../img/img_TaskC/Validation_target_dist.png" width="45%" alt="Validation Target Distribution"/>
  <img src="../../img/img_TaskC/Test_Sample_target_dist.png" width="45%" alt="Test Target Distribution"/>
</div>

<br>

**2. Code Length Distribution (Cleaned)**
Analysis of snippet lengths (excluding extreme outliers). In "Mixed-Source" tasks, length can correlate with the likelihood of AI intervention (AIs tend to refactor concisely or verbosely depending on the prompt).

<div style="text-align:center">
  <img src="../../img/img_TaskC/Train_length_dist.png" width="45%" alt="Train Length Distribution"/>
  <img src="../../img/img_TaskC/Validation_length_dist.png" width="45%" alt="Validation Length Distribution"/>
  <img src="../../img/img_TaskC/Test_Sample_length_dist.png" width="45%" alt="Test Length Distribution"/>
</div>

<br>

**3. Predominant Languages**
Overview of the languages involved in the hybrid modification/generation task.

<div style="text-align:center">
  <img src="../../img/img_TaskC/Train_languages.png" width="45%" alt="Train Languages"/>
  <img src="../../img/img_TaskC/Validation_languages.png" width="45%" alt="Validation Languages"/>
  <img src="../../img/img_TaskC/Test_Sample_languages.png" width="45%" alt="Test Languages"/>
</div>

This information helps to define:

- Whether to treat the problem as **Classification** or **Regression**.
- How to manage **context length** in Transformer models (e.g., very long snippets might require sliding windows).
- The **Data Augmentation** strategy needed to cover less represented languages.

---


## 🚀 Execution Instructions

### 1. Training

To start the training pipeline with logging to console, TensorBoard, and CometML:
```bash
python -m src.src_TaskC.train
```

The output will include a progress bar with real-time metrics. The best model (based on Macro-F1) will be automatically saved in `results/results_TaskC/checkpoints/`.

### 2. Inference and Submission

Per generare il file `submission_task_c.csv` file for the leaderboard:
```bash
python -m src.src_TaskC.generate_submission
```
The script automatically detects the `test.parquet` file (searching also within Kaggle download subfolders) and generates the file in `results/results_TaskC/submission/submission_task_c.csv`.

---

## 📊 Repository Structure Sub Task-C

```bash
├── 📁 src
│   └── 📁 src_TaskC
│       ├── 📁 config
│       │   └── ⚙️ config.yaml
│       │
│       ├── 📁 dataset
│       │   ├── 🐍 Inference_dataset.py
│       │   └── 🐍 dataset.py
│       │
│       ├── 📁 models
│       │   └── 🐍 model.py
│       │
│       ├── 📁 utils
│       │   └── 🐍 utils.py
│       │
│       ├── 📝 README.md
│       │
│       ├── 🐍 generate_submission.py
│       ├── 🐍 inference.py
│       │
│       └── 🐍 train.py
```

---

> [!CAUTION]
> README ANCORA IN FASE DI SVILUPPO...

---

<!--───────────────────────────────────────────────-->
<!--                   AUTORE                     -->
<!--───────────────────────────────────────────────-->

<div align="center">
  <h2>✨ Autore ✨</h2>

  <p>
    <strong>Giovanni Giuseppe Iacuzzo</strong><br>
    <em>AI & Cybersecurity Engineering Student</em><br>
    <em>University of Kore, Enna</em>
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
