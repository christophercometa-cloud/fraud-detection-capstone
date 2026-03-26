# 💳 Credit Card Fraud Detection: An End-to-End Machine Learning Pipeline

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Deployed-009688)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
Credit Card Fraud has cost banks over $billions per year. Conventional Rules-Based Fraud Detection Systems are normally unable to recognize new types of fraud that continue to evolve. These traditional systems will also generate far too many False Positives when reviewing legitimate customer transactions. The result is serious customer dissatisfaction as well as loss of potential sales/revenue.

This Capstone Project describes the entire Machine Learning Lifecycle for Real-Time prediction of potential fraudulent transactions. It uses an XGBoost Classifier optimized specifically for very unbalanced data. As such, the classifier maximizes the ability to detect all potential fraudulent transactions while minimizing the number of false positives. Additionally, the system utilizes a Generative Artificial Intelligence ("AI") "Copilot" which can create plain English descriptions of each model decision so that Human Review Teams may understand the reasoning behind them.

---

## 📊 The Dataset & Dictionary
The dataset used for this project is the widely recognized [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Due to the nature of financial data, the core features (`V1` through `V28`) are the result of a Principal Component Analysis (PCA) transformation to protect user privacy.

* **Total Transactions:** 284,807
* **Fraudulent Transactions:** 492
* **Imbalance Ratio:** ~0.17% of transactions are fraudulent.

| Variable | Type | Description |
| :--- | :--- | :--- |
| `Time` | Float | Seconds elapsed between this transaction and the first transaction. |
| `V1` - `V28` | Float | Principal components obtained via PCA (anonymized). |
| `Amount` | Float | Transaction amount in USD. |
| `Class` (Target)| Integer | Indicates whether the transaction is fraud (1) or legitimate (0). |

---

## 🏗️ Repository Structure
```text
fraud-detection-capstone/
├── data/
│   ├── credicard.csv         # Original data 
│   └── processed/            # Cleaned data (X_train_res.csv, etc.)
├── notebooks/
│   ├── 01_EDA_and_Cleaning.ipynb          
│   └── 02_Modeling_and_Evaluation.ipynb     
├── src/
│   ├── data_prep.py          # Data cleaning, scaling, and SMOTE script
│   ├── train.py              # XGBoost training script
│   └── app.py                # FastAPI deployment script
├── models/
│   ├── xgb_fraud_model.pkl   # Saved model artifact
│   └── robust_scaler.pkl     # Saved scaler artifact
├── presentations/
│   ├── Credit_Card_Fraud_Detection_Technical_Presentation.pdf    
│   └── Next-Generation_Fraud_Detection_Business_Presentation.pdf     
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation