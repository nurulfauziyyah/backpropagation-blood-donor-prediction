# 🩸 Active Blood Donor Prediction using Backpropagation Neural Network

---

## 📌 Overview

This project implements a **Backpropagation-based Artificial Neural Network (ANN)** to predict active blood donors using data from the Blood Transfusion Service Center.

The problem is formulated as a **binary classification task**, where the objective is to classify whether a donor will donate blood again within a specified period.

To address class imbalance, two resampling techniques were compared:

- ROSE (Random Over Sampling Examples)
- SMOTE (Synthetic Minority Over-sampling Technique)

---

## 🎯 Objective

To develop a neural network model capable of predicting active donors and evaluate the impact of data balancing techniques on classification performance.

---

## 📊 Dataset Description

- Total observations: 748
- Features: 4 input variables
- Target: 1 binary variable

### Feature Variables

| Variable | Type    | Description |
|----------|---------|------------|
| Recency  | Integer | Months since last donation |
| Frequency | Integer | Total number of blood donations |
| Monetary | Integer | Total blood donated (cc) |
| Time     | Integer | Months since first donation |

### Target Variable

| Variable | Type   | Description |
|----------|--------|------------|
| Donated_Blood | Binary | 1 = Donated in March 2007, 0 = Did not donate |

The dataset exhibits class imbalance, motivating the use of resampling strategies.

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Feature normalization
- Train-test split
- Class imbalance handling using ROSE and SMOTE

### 2️⃣ Model Architecture

- Artificial Neural Network (ANN)
- Supervised learning
- Backpropagation algorithm
- Binary classification output

### 3️⃣ Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- AUC (ROC Curve)

---

## 📈 Performance Comparison

| Metric | ROSE | SMOTE |
|--------|------|--------|
| Accuracy | 70% | 68% |
| Precision | 41.7% | 39.3% |
| Recall | 71.4% | 68.6% |
| F1-Score | 52.6% | 50% |
| AUC | 0.802 | 0.761 |

### Training Behavior

- ROSE:
  - Accuracy stabilized above 0.95 in mid-epoch
  - Loss converged faster and more smoothly

- SMOTE:
  - Accuracy stabilized around 0.72
  - Loss stable but slightly more fluctuative

---

## 🏆 Best Model

The ANN model trained using **ROSE-balanced data** achieved the best performance:

- Accuracy: 70%
- F1-Score: 52.6%
- AUC: 0.802

This indicates that ROSE provided better class distribution learning for this dataset.

---

## 🔎 Key Insights

- Neural networks can effectively capture behavioral donation patterns.
- Handling class imbalance significantly impacts model performance.
- ROSE outperformed SMOTE in this specific dataset.

---

## 🛠 Tool
RStudio
