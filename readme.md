# 💳 Credit Card Fraud Detection: An End-to-End Machine Learning Pipeline

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Deployed-009688)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
Financial institutions lose billions of dollars annually to credit card fraud. Traditional, rules-based fraud detection systems often struggle to keep up with evolving fraud tactics and tend to flag too many legitimate transactions (False Positives), resulting in severe customer friction and lost revenue.

This capstone project demonstrates an end-to-end machine learning lifecycle to predict fraudulent transactions in real-time. By leveraging an XGBoost classifier optimized for highly imbalanced data, this solution maximizes fraud detection while minimizing false positives. Additionally, it integrates a Generative AI "Copilot" to provide plain-English explanations of model decisions for human review teams.

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
│   ├── Technical_Presentation.pdf    
│   └── Business_Presentation.pdf     
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation