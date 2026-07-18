# ECG Arrhythmia Classification using a Hybrid Autoencoder-CNN

> A research-oriented implementation and educational reproduction of the IEEE paper:
> **"Hybrid Autoencoder-CNN Model for Accurate ECG Arrhythmia Classification Using the MIT-BIH Dataset" (IEEE GCWCN 2025)**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Research](https://img.shields.io/badge/Type-Research%20Project-red)

---

# Overview

This repository documents the complete implementation and educational reproduction of an IEEE research paper on **ECG Arrhythmia Classification** using a **Hybrid Autoencoder-CNN** architecture.

Unlike many repositories that only provide source code, this project is designed as an **educational research notebook**.

Every implementation step is accompanied by explanations covering:

- Biomedical signal processing concepts
- ECG fundamentals
- Dataset exploration
- Signal preprocessing
- Deep learning implementation
- Engineering decisions
- Paper reproduction

The objective is not only to reproduce the reported results, but also to understand every stage of the pipeline.

---

# Project Objectives

- Reproduce the proposed IEEE methodology
- Understand every preprocessing step
- Learn Biomedical Signal Processing (BSP)
- Build reusable Python modules
- Develop a professional research-oriented GitHub repository
- Follow software engineering best practices

---

# Paper Information

**Title**

Hybrid Autoencoder-CNN Model for Accurate ECG Arrhythmia Classification Using the MIT-BIH Dataset

**Conference**

IEEE GCWCN 2025

**Problem Domain**

Biomedical Signal Processing (BSP)

ECG Arrhythmia Classification

Deep Learning

Computer-Aided Diagnosis (CAD)

---

# Current Progress

Project Status:

- [x] Project initialized
- [x] Repository structure created
- [x] Python environment configured
- [x] MIT-BIH dataset integrated
- [x] WFDB data loader implemented
- [x] First ECG signal visualization
- [ ] Exploratory Data Analysis
- [ ] Signal preprocessing
- [ ] Heartbeat segmentation
- [ ] Autoencoder
- [ ] CNN classifier
- [ ] Hybrid model
- [ ] Evaluation
- [ ] Paper reproduction

---

# Repository Structure

```text
ECG-Autoencoder-CNN/

├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── docs/
├── models/
├── notebooks/
│   └── ECG_Arrhythmia_Classification.ipynb
│
├── outputs/
├── src/
│   └── dataset.py
│
├── tests/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Dataset

This project uses the **MIT-BIH Arrhythmia Database**.

Official Source:

https://physionet.org/content/mitdb/

After downloading the dataset, extract all files into:

```text
data/raw/
```

Expected structure:

```text
data/raw/

100.dat
100.hea
100.atr

101.dat
101.hea
101.atr

...
```

The dataset is **not included** in this repository.

---

# Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn
- WFDB
- Jupyter Notebook

Future stages:

- NeuroKit2
- TensorFlow / Keras
- PyTorch (comparison)
- BioSPPy

---

# Educational Notebook

The notebook is developed progressively.

Each section includes

- theoretical explanation
- engineering discussion
- implementation
- visualization
- interpretation

The notebook is intended to serve as both

- project implementation
- learning material

---

# Engineering Philosophy

This repository follows the principles of

- Reproducible Research
- Research Engineering
- Modular Software Design
- Conventional Commits
- Clean Code
- Educational Documentation

---

# Git Workflow

The project uses Conventional Commits.

Examples:

```text
feat(dataset): implement MIT-BIH loader

feat(eda): add annotation visualization

docs: update README

fix(preprocessing): correct normalization

refactor(dataset): improve loader interface
```

---

# Planned Modules

```
src/

dataset.py

eda.py

preprocessing.py

segmentation.py

features.py

autoencoder.py

cnn.py

train.py

evaluate.py

utils.py
```

---

# Future Improvements

- Kalman filtering
- Heartbeat segmentation
- Autoencoder implementation
- CNN implementation
- Hybrid architecture
- Model evaluation
- Confusion matrix
- TensorBoard integration
- Experiment tracking
- Docker support

---

# Author

Mahdi Shafighi

M.Sc. Student in Artificial Intelligence

University Project — Biomedical Signal Processing

---

# Acknowledgment

This repository is developed for educational and research purposes.

Special thanks to

- PhysioNet
- WFDB Developers
- IEEE Authors