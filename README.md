# ECG Arrhythmia Classification — Autoencoder, CNN & Hybrid Models

> A research-oriented and reproducible study of ECG arrhythmia classification using the MIT-BIH Arrhythmia Database.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange)
![Research](https://img.shields.io/badge/Type-Research%20Project-red)

## Overview

This project builds a modular pipeline for ECG signal processing and arrhythmia classification. The experimental plan compares three model families:

1. **CNN baseline**
2. **Autoencoder-based representation learning**
3. **Hybrid Autoencoder + CNN model**

The repository is designed to make preprocessing, training, evaluation, and experiment artifacts traceable and reproducible.

## Dataset

The experiments use the **MIT-BIH Arrhythmia Database** from PhysioNet. The dataset is not included in this repository.

Place the downloaded records under:

```text
data/raw/
```

## Pipeline

```text
MIT-BIH
   ↓
WFDB Loading
   ↓
Signal Validation
   ↓
Preprocessing
   ↓
Heartbeat Segmentation
   ↓
┌───────────────┬──────────────────┬─────────────────┐
│ CNN Baseline  │ Autoencoder      │ Hybrid AE + CNN │
└───────────────┴──────────────────┴─────────────────┘
   ↓
Evaluation
   ↓
Metrics + Confusion Matrix + Artifacts
```

## Evaluation Plan

The experiment layer is intended to report:

- Accuracy
- Precision
- Recall / Sensitivity
- F1-score
- Class-wise precision, recall, and F1
- Confusion matrix
- Model comparison
- Training history

### Research Integrity

**No final performance percentage is claimed until the experiment is actually executed on the MIT-BIH data.** This repository separates implemented infrastructure from verified experimental results.

## Current Status

**Active development.**

Implemented infrastructure includes the project structure, dataset integration direction, modular source layout, tests, and CI foundations. The complete experimental comparison remains dependent on executing the pipeline with the real dataset.

## Planned Research Milestones

- [x] Repository and modular architecture
- [x] MIT-BIH data-loading foundation
- [x] Validation and test infrastructure
- [x] CI foundation
- [ ] End-to-end MIT-BIH experiment
- [ ] CNN baseline training
- [ ] Autoencoder training
- [ ] Hybrid model training
- [ ] Final comparative evaluation
- [ ] Reproducible experiment artifacts

## Repository Structure

```text
ECG-Autoencoder-CNN/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
├── models/
├── notebooks/
├── outputs/
├── src/
├── tests/
├── requirements.txt
└── README.md
```

## Engineering Principles

- Modular design
- Reproducible research
- Explicit experiment configuration
- Automated testing
- Continuous Integration
- No unverified claims

## Technology

Python • NumPy • Pandas • SciPy • Scikit-learn • WFDB • TensorFlow/Keras • Matplotlib

## Author

Mohammad Mahdi Shafighi — M.Sc. Artificial Intelligence
