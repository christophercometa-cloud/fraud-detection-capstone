import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc, confusion_matrix

def load_processed_data(data_dir):
    """Loads the split and resampled datasets."""
    print("Loading processed training and testing data...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train_res.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train_res.csv')).squeeze() # squeeze to make it a Series
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).squeeze()
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Initializes and trains the XGBoost classifier."""
    print("Training the XGBoost model (this might take a minute)...")
    
    # Initialize XGBoost. 
    # scale_pos_weight is another way to handle imbalance if we didn't use SMOTE, 
    # but since we used SMOTE, standard parameters work well.
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    xgb_model.fit(X_train, y_train)
    return xgb_model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model using metrics suited for highly imbalanced data."""
    print("\nEvaluating model performance...")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 1. Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # 2. Confusion Matrix
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    # 3. Precision-Recall AUC (Crucial for fraud detection)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    print(f"\nArea Under the Precision-Recall Curve (PR-AUC): {pr_auc:.4f}")

if __name__ == "__main__":
    # Define paths relative to this file so execution cwd does not matter
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    model_dir = os.path.join(project_root, 'models')
    
    # Execute the training pipeline
    try:
        X_train, X_test, y_train, y_test = load_processed_data(processed_data_dir)
        
        trained_model = train_model(X_train, y_train)
        
        evaluate_model(trained_model, X_test, y_test)
        
        # Save the trained model
        print("\nSaving the trained model artifact...")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(trained_model, os.path.join(model_dir, 'xgb_fraud_model.pkl'))
        print("Model saved successfully! 🚀")
        
    except FileNotFoundError as err:
        print("Error: Processed data files not found. Please run 'python src/data_prep.py' first.")
        print(f"Missing path details: {err}")