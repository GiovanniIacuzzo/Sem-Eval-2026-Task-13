# SemEval-2026 Task 13: Subtask A - Machine-Generated Code Detection
## 📌 Subtask A Objective

<div align="center">
  <a href="README.it.md">
    <img src="https://img.shields.io/badge/Lingua-Italiano-008C45?style=for-the-badge&logo=italian&logoColor=white" alt="Leggi in Italiano">
  </a>
</div>

**Subtask A** of the SemEval-2026 Task 13 challenge consists of building a **binary classification** model capable of distinguishing **machine-generated** code from **human-written** code.

- **Labels:** - `0` = machine-generated code  
  - `1` = human-written code
- **Training Languages:** C++, Python, Java  
- **Training Domain:** Algorithmic (e.g., LeetCode-style problems)

The goal is to evaluate the model's ability to **generalize** to languages or domains **not seen during training**.

| Setting                              | Languages              | Domain                 |
|--------------------------------------|-----------------------|------------------------|
| Seen Languages & Seen Domains         | C++, Python, Java     | Algorithmic            |
| Unseen Languages & Seen Domains       | Go, PHP, C#, C, JS    | Algorithmic            |
| Seen Languages & Unseen Domains       | C++, Python, Java     | Research, Production   |
| Unseen Languages & Domains            | Go, PHP, C#, C, JS    | Research, Production   |

---

## 📝 Initial Dataset Analysis

To better understand the available data, an `info_dataset.py` script was created that:

1. Loads the `.parquet` files for Subtask A (train, validation, test).  
2. Calculates statistics on code snippets: length, distribution by language, and by label.  
3. Saves visualizations in the `img` folder for a quick overview of the data.

---

### Examples of results saved in `img`:

Distribution and statistics for Train, Validation, and Test datasets:

<div style="text-align:center">
  <img src="../../img/img_TaskA/Train_length_label.png" width="30%" />
  <img src="../../img/img_TaskA/Validation_length_label.png" width="30%" />
  <img src="../../img/img_TaskA/Test_length_label.png" width="30%" />
</div>

<div style="text-align:center">
  <img src="../../img/img_TaskA/Train_label_language.png" width="30%" />
  <img src="../../img/img_TaskA/Validation_label_language.png" width="30%" />
  <img src="../../img/img_TaskA/Test_label_language.png" width="30%" />
</div>

<div style="text-align:center">
  <img src="../../img/img_TaskA/Train_top_generators.png" width="30%" />
  <img src="../../img/img_TaskA/Validation_top_generators.png" width="30%" />
  <img src="../../img/img_TaskA/Test_top_generators.png" width="30%" />
</div>

This information helps to understand:

- The predominance of Python in the dataset.
- The relative imbalance between human and generated snippets.
- The general characteristics of the most common generators.

---

## 🚀 Execution Instructions

### 1. Training
To start the training pipeline with logging to console, TensorBoard, and CometML:
```bash
python -m src.src_TaskA.train
```

The output will include a progress bar with real-time metrics. The best model (based on Macro-F1) will be automatically saved in `results/results_TaskA/checkpoints/`.

### 2. Inference and Submission

To generate the valid `submission_task_a.csv` file for the leaderboard:
```bash
python -m src.src_TaskA.generate_submission
```
The script automatically detects the `test.parquet` file (searching also within Kaggle download subfolders) and generates the file in `results/results_TaskA/submission/submission_task_a.csv`.

---

## 📊 Repository Structure Sub Task-A

```bash
├── 📁 src
│   └── 📁 src_TaskA
│       ├── 📁 config
│       │   └── ⚙️ config.yaml
│       │
│       ├── 📁 dataset
│       │   ├── 🐍 Inference_dataset.py
│       │   └── 🐍 dataset.py
│       │
│       ├── 📁 features
│       │   └── 🐍 stylometry.py
│       │
│       ├── 📁 models
│       │   └── 🐍 model.py
│       │
│       ├── 📁 scripts
│       │   ├── 🐍 augment_data.py
│       │   └── 🐍 debug_data.py
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

## 🧠 Architettura del Modello e Strategia

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