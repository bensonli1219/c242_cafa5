# CAFA5 Protein Function Prediction (C242 Project)

This project focuses on predicting Gene Ontology (GO) labels from protein sequences using machine learning models.

---

## 📌 Project Overview

This project builds a complete pipeline including:

- Data preprocessing and splitting (70/15/15)
- Feature engineering (K-mer and ESM embeddings)
- Model training (MLP)
- Hyperparameter tuning
- GO namespace-specific modeling (BPO / CCO / MFO)
- Evaluation using IA-weighted Fmax

---

## 📁 Project Structure

.
├── kmer/
│   ├── data_preprocessing_kmer_70.ipynb
│   ├── MLP_kmer.ipynb
│   ├── kmer_tuning.ipynb
│   ├── kmer_GOsplit_70.ipynb
│   ├── kmer_error_analysis.ipynb
│
├── ESM/
│   ├── ESM2_embedding.ipynb
│   ├── MLP_ESM.ipynb
│   ├── ESM_tuning.ipynb
│   ├── ESM_GOsplit_70.ipynb
│   ├── ESM_error_analysis.ipynb
│
├── cafa-5-protein-function-prediction
├── requirements.txt
└── README.md

cafa-5-protein-function-prediction include raw data
---

## 🚀 How to Run (Step-by-Step)

---

## 🔵 K-mer Pipeline

### 1️⃣ Data Preprocessing

Run:
kmer/data_preprocessing_kmer_70.ipynb

This step:
- Cleans protein sequences
- Converts GO labels into multi-label format
- Filters GO terms (frequency ≥ 20)
- Splits data into:
  - 70% training
  - 15% validation
  - 15% test
- Saves processed datasets

---

### 2️⃣ Baseline Model (All GO)

Run:
kmer/MLP_kmer.ipynb

This step:
- Uses K-mer features (local amino acid patterns)
- Trains an MLP model on all GO labels
- Provides initial baseline performance

---

### 3️⃣ Hyperparameter Tuning

Run:
kmer/kmer_tuning.ipynb

This step:
- Tunes model parameters (hidden size, dropout, etc.)
- Selects the best model using validation set

---

### 4️⃣ GO Namespace-specific Training

Run:
kmer/kmer_GOsplit_70.ipynb

This step:
- Splits GO labels into:
  - BPO (Biological Process)
  - CCO (Cellular Component)
  - MFO (Molecular Function)
- Trains separate models for each namespace
- Uses:
  - Training set → model learning
  - Validation set → model selection
  - Test set → final evaluation

---

### 5️⃣ Error Analysis

Run:
kmer/kmer_error_analysis.ipynb

This step:
- Analyzes incorrect predictions
- Identifies model weaknesses
- Examines label imbalance and difficulty

---

## 🔵 ESM Pipeline

### 6️⃣ Generate ESM2 Embeddings

Run:
ESM/ESM2_embedding.ipynb

This step:
- Uses pretrained ESM2 model
- Converts protein sequences into fixed-length embeddings
- Captures global sequence context

---

### 7️⃣ Baseline Model (ESM Features)

Run:
ESM/MLP_ESM.ipynb

This step:
- Uses ESM embeddings as input features
- Trains an MLP model
- Provides baseline performance using deep representations

---

### 8️⃣ Hyperparameter Tuning (ESM)

Run:
ESM/ESM_tuning.ipynb

This step:
- Tunes model parameters
- Selects best configuration using validation performance

---

### 9️⃣ GO Namespace-specific Training (ESM)

Run:
ESM/ESM_GOsplit_70.ipynb

This step:
- Splits GO labels into BPO / CCO / MFO
- Trains separate models for each namespace
- Evaluates final performance on test set

---

### 🔟 Error Analysis (ESM)

Run:
ESM/ESM_error_analysis.ipynb

This step:
- Analyzes prediction errors
- Compares performance with K-mer
- Shows advantages of deep embeddings

---

## 📊 Evaluation Metric: IA-weighted Fmax

- Fmax = best F1 score across all thresholds
- IA-weighted = each GO term weighted by information content
- Rare and specific GO terms have higher importance

---

## ⚠️ Notes

- Large files (.npy, .npz, .pt) are not included
- Please regenerate data using preprocessing notebooks
- Training was performed on HPC (Savio)

---

## 👤 Author

Yu-Shan Fu
UC Berkeley Bioengineering
