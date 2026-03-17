#author: Christopher Cometa
#date: 2026-03-15
#description: This script prepares the credit card fraud dataset by cleaning, scaling, and applying SMOTE to handle class imbalance. It saves the processed datasets and the fitted scaler for later use in model training and deployment.

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    """Loads the raw dataset."""
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def clean_and_scale(df):
    """Handles duplicates and scales Time/Amount features."""
    print("Cleaning and scaling data...")
    
    # The Kaggle dataset contains about 1081 duplicates; it's best practice to drop them
    df = df.drop_duplicates()
    
    # Initialize scaler
    scaler = RobustScaler()
    
    # Scale Amount and Time (RobustScaler is less prone to extreme outliers)
    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_Scaled'] = scaler.fit_transform(df[['Time']])
    
    # Drop original columns
    df = df.drop(['Time', 'Amount'], axis=1)
    
    return df, scaler

def split_and_resample(df, target_col='Class'):
    """Splits the data and applies SMOTE to the training set only."""
    print("Splitting data and applying SMOTE...")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Stratified split to maintain the 0.17% fraud ratio in the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE strictly to the training data to avoid data leakage
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    return X_train_res, X_test, y_train_res, y_test

if __name__ == "__main__":
    # 1. Define paths relative to this file so execution cwd does not matter
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_candidates = [
        os.path.join(project_root, 'data', 'raw', 'creditcard.csv'),
        os.path.join(project_root, 'data', 'creditcard.csv'),
    ]
    input_path = next((p for p in input_candidates if os.path.exists(p)), None)
    if input_path is None:
        raise FileNotFoundError(
            "Could not find creditcard.csv in expected locations: "
            + ", ".join(input_candidates)
        )

    output_dir = os.path.join(project_root, 'data', 'processed')
    model_dir = os.path.join(project_root, 'models')
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 2. Execute pipeline
    df_raw = load_data(input_path)
    df_clean, fitted_scaler = clean_and_scale(df_raw)
    X_train_res, X_test, y_train_res, y_test = split_and_resample(df_clean)
    
    # 3. Save processed data for the train.py script
    print("Saving processed datasets...")
    X_train_res.to_csv(os.path.join(output_dir, 'X_train_res.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train_res.to_csv(os.path.join(output_dir, 'y_train_res.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    # 4. Save the scaler artifact for the deployment API
    print("Saving scaler artifact...")
    joblib.dump(fitted_scaler, os.path.join(model_dir, 'robust_scaler.pkl'))
    
    print("Data preparation complete! 🎉")