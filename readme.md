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
```

## 🚀 Local Deployment Guide

Follow these steps to deploy the Fraud Detection API and the Generative AI Copilot on your local machine.

### Prerequisites
* Python 3.9+
* Git (and Git LFS if you are downloading the demo video)
* A free [Google Gemini API Key](https://aistudio.google.com/)

### 1. Clone the Repository
Open your terminal and clone this project to your local machine:
```bash
git clone https://github.com/christophercometa-cloud/fraud-detection-capstone.git
cd fraud-detection-capstone
```
### 2. Create a Virtual Environment
It is highly recommended to run this project in an isolated virtual environment to prevent dependency conflicts.

Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
Install all required Python packages (FastAPI, XGBoost, Google Generative AI SDK, etc.):
```bash
pip install fastapi uvicorn pandas scikit-learn xgboost pydantic joblib google-generativeai
```
(Note: If you are using the provided requirements file, simply run pip install -r requirements.txt)

### 4. Set Up the Generative AI Copilot (Step 9)
To enable the LLM explanations, you must provide your Google Gemini API key as an environment variable in your terminal.

Mac/Linux:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

Windows (Command Prompt):
```bash
set GEMINI_API_KEY="your_api_key_here"
```

Windows (PowerShell):
```bash
$env:GEMINI_API_KEY="your_api_key_here"
```

### 5. Start the Server
Navigate to the directory containing app.py and start the Uvicorn web server:
```bash
uvicorn app:app --reload
```

### 6. Test the API via Swagger UI
Once the terminal displays Application startup complete, open your web browser and navigate to the interactive API documentation:
👉 https://www.google.com/search?q=http://127.0.0.1:8000/docs

Click on the green POST /predict_and_explain endpoint.

Click "Try it out".

Paste a transaction array into the Request Body.

Hit Execute to see the XGBoost model and GenAI Copilot in action!
