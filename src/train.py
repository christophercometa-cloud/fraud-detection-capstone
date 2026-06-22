#Author: Christopher Cometa
#Date: 2026-03-15
#Description: This script trains an XGBoost model on the processed credit card fraud dataset. It evaluates the model using classification metrics suitable for imbalanced data and saves the trained model for later use

from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc, confusion_matrix
def load_processed_data(data_dir):
    """Loads the split and resampled datasets."""
    print("Loading processed training and testing data...")
    X_train = pd.read_csv(data_dir / 'X_train_res.csv')
    X_test = pd.read_csv(data_dir / 'X_test.csv')
    y_train = pd.read_csv(data_dir / 'y_train_res.csv').squeeze() # squeeze to make it a Series
    y_test = pd.read_csv(data_dir / 'y_test.csv').squeeze()
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using metrics suited for highly imbalanced data.
    Returns the PR-AUC score for comparison.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 1. Classification Report
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # 2. Confusion Matrix
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    # 3. Precision-Recall AUC (The most crucial metric for fraud detection)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    print(f"Area Under the Precision-Recall Curve (PR-AUC): {pr_auc:.4f}")
    
    return pr_auc

if __name__ == "__main__":
    # Define paths relative to this file so execution cwd does not matter
    project_root = Path(__file__).resolve().parent.parent
    processed_data_dir = project_root / 'data' / 'processed'
    model_dir = project_root / 'models'

    try:
        X_train, X_test, y_train, y_test = load_processed_data(processed_data_dir)
        
        # Define the models we want to train and compare
        models_to_train = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
        }
        
        best_model = None
        best_pr_auc = -1
        best_model_name = ""
        
        # Loop through, train, and evaluate each model
        for name, model in models_to_train.items():
            print(f"\n{'='*15} Training {name} {'='*15}")
            model.fit(X_train, y_train)
            
            print(f"\n--- Evaluating {name} ---")
            pr_auc = evaluate_model(model, X_test, y_test)
            
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_model = model
                best_model_name = name
        
        # Save the best performing model
        print(f"\n{'='*50}")
        print(f"🏆 Best performing model: {best_model_name} with PR-AUC of {best_pr_auc:.4f}")
        print(f"Saving {best_model_name} as the final model artifact...")
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_dir / 'xgb_fraud_model.pkl')
        print("Model saved successfully! 🚀")
        
    except FileNotFoundError as err:
        print("Error: Processed data files not found. Please run 'python src/data_prep.py' first.")
        print(f"Missing path details: {err}")