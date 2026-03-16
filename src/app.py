from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import uvicorn
from pathlib import Path

# 1. Initialize the application
app = FastAPI(title="Real-Time Fraud Detection API", 
              description="Predicts whether a credit card transaction is fraudulent.")

# 2. Load the trained model (Ensure the path matches your project structure)
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / 'models' / 'xgb_fraud_model.pkl'

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    model = None

# 3. Define the input data schema using Pydantic
# For brevity, we accept a list of features, but in a real app, 
# you'd map out all 30 features (V1-V28, Amount_Scaled, Time_Scaled)
class Transaction(BaseModel):
    features: list[float]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [0.1, -1.2, 3.4, 0.5, 1.1, -0.9, 0.0, 2.1,
                             -1.5, 0.3, 0.1, 1.2, -0.4, 0.8, 1.1, -0.2,
                             2.3, 0.4, -1.1, 0.5, -0.1, 0.2, 1.5, -0.8,
                             0.9, 1.2, -0.3, 0.1, 0.75, -0.25]
            }
        }
    )

# 4. Create the prediction endpoint
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    # Ensure the correct number of features are provided (30 in this case)
    if len(transaction.features) != 30:
        raise HTTPException(status_code=400, detail="Exactly 30 features are required.")
    
    # Convert input to a DataFrame/2D array for the model
    input_data = pd.DataFrame([transaction.features])
    
    # Make prediction and get probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": float(probability),
        "status": "Transaction Blocked" if prediction == 1 else "Transaction Approved"
    }

# Run this script using: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)