from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc
import joblib
from pathlib import Path

from data_prep import X_test, X_train_res, y_train_res, y_test

# 1. Train XGBoost
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_res, y_train_res)

# 2. Evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Calculate PR-AUC
precision, recall, _ = precision_recall_curve(y_test, y_prob_xgb)
pr_auc = auc(recall, precision)
print(f"XGBoost PR-AUC: {pr_auc:.4f}")

# 3. Save Artifacts for Reproducibility
models_dir = Path(__file__).resolve().parent.parent / 'models'
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / 'xgb_fraud_model.pkl'
joblib.dump(xgb_model, model_path)