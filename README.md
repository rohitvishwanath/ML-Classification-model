# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning project to detect fraudulent credit card transactions using imbalanced data handling techniques and deployed with an interactive web UI.

---

## 🚀 Features

- 🔍 Real-time fraud prediction
- 📊 Batch prediction using CSV upload
- ⚖️ Handles imbalanced data using SMOTE
- 📈 Model evaluation using Precision, Recall, F1-score
- 🎯 Hyperparameter tuning with GridSearchCV
- 🌐 Interactive UI built with Streamlit

---

## 🧠 Problem Statement

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions are rare compared to normal transactions. The goal is to accurately identify fraud cases while minimizing false negatives.

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Imbalanced-learn (SMOTE)
- Streamlit

---

## 📊 Machine Learning Workflow

1. Data Preprocessing
   - Handling missing values
   - Feature scaling (Standardization)
   - Outlier detection

2. Model Building
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest

3. Handling Imbalance
   - SMOTE (Synthetic Minority Oversampling Technique)
   - Random Oversampling & Undersampling

4. Model Evaluation
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

5. Model Optimization
   - GridSearchCV
   - K-Fold Cross Validation

---

## 📈 Results

- Achieved high accuracy with strong recall for fraud detection
- Observed that SMOTE does not always improve performance depending on dataset characteristics
- Random Forest performed best among all models

---

## 🖥️ Application (UI)

The project includes a Streamlit-based web app:
- Manual input prediction
- CSV batch prediction
- Fraud probability visualization

---

## 📦 Installation

```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
streamlit run app.py
