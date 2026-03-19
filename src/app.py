#author: Christopher Cometa
#date: 2026-03-15
#description: This script sets up a FastAPI application that serves a trained XGBoost model for real-time fraud detection. It defines an endpoint that accepts transaction data, processes it, and returns
# a prediction along with the probability of fraud and a status message indicating whether the transaction is approved or blocked.


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import uvicorn
from pathlib import Path
import json
import os
import socket
from urllib import error, parse, request

# 1. Initialize the application
app = FastAPI(title="Real-Time Fraud Detection API", 
              description="Predicts whether a credit card transaction is fraudulent.")

# 2. Load the trained model
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / 'models' / 'xgb_fraud_model.pkl'
default_feature_names = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount_Scaled', 'Time_Scaled'
]

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    model = None

if model is not None and hasattr(model, 'feature_names_in_'):
    feature_names = list(model.feature_names_in_)
else:
    feature_names = default_feature_names

# --- 3. Setup Gemini API  ---
# Gemini is called over REST here to avoid the deprecated SDK's gRPC DNS issues.
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "15"))
EXPOSE_GENAI_ERRORS = os.getenv("EXPOSE_GENAI_ERRORS", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
selected_model = None


def _select_available_gemini_model() -> str | None:
    """Pick a currently available Gemini model that supports generateContent."""
    preferred_models = [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-lite",
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-flash-8b",
    ]

    try:
        response_body = _gemini_request("/models")
        available = response_body.get("models", [])
    except RuntimeError:
        return None

    generation_capable = {
        m.get("name")
        for m in available
        if "generateContent" in m.get("supportedGenerationMethods", [])
    }
    if not generation_capable:
        return None

    for name in preferred_models:
        if name in generation_capable:
            return name

    return sorted(generation_capable)[0]


def _gemini_request(path: str, payload: dict | None = None) -> dict:
    query = parse.urlencode({"key": GOOGLE_API_KEY})
    url = f"{GEMINI_API_BASE_URL}{path}?{query}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )

    try:
        with request.urlopen(req, timeout=GEMINI_TIMEOUT_SECONDS) as http_response:
            return json.loads(http_response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {exc.code}: {details}") from exc
    except (error.URLError, TimeoutError, socket.timeout, socket.gaierror) as exc:
        raise RuntimeError(f"Gemini network error: {exc}") from exc


def _get_gemini_model() -> str | None:
    global selected_model

    if selected_model is None and GOOGLE_API_KEY:
        selected_model = _select_available_gemini_model()
    return selected_model


def _fallback_explanation(probability: float) -> str:
    confidence_pct = round(probability * 100, 2)
    return (
        f"Transaction blocked because the fraud model detected patterns strongly associated "
        f"with charge abuse and assigned a {confidence_pct}% fraud probability. Place an "
        f"immediate temporary hold on the account and verify the activity directly with the cardholder."
    )


def _build_fraud_prompt(probability: float) -> str:
    return f"""
    You are a fraud analyst copilot. A transaction was just blocked by our XGBoost model.
    The model is {round(probability * 100, 2)}% confident this is fraud.
    Write a concise, 2-sentence explanation for a human analyst explaining that the transaction
    was blocked due to anomalous patterns. Recommend an immediate account freeze.
    """.strip()


def _generate_fraud_explanation(probability: float) -> str:
    model_name = _get_gemini_model()
    if not GOOGLE_API_KEY or model_name is None:
        raise RuntimeError("No Gemini API key or compatible model available.")

    prompt = _build_fraud_prompt(probability)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 120,
        },
    }
    encoded_model = parse.quote(model_name, safe="/")
    response_body = _gemini_request(f"/{encoded_model}:generateContent", payload)

    candidates = response_body.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        text_parts = [
            part.get("text", "").strip()
            for part in content.get("parts", [])
            if part.get("text", "").strip()
        ]
        if text_parts:
            explanation_text = ' '.join(text_parts)
            return f"GenAI Copilot: {explanation_text}"

    raise RuntimeError("Gemini returned no text content.")


if GOOGLE_API_KEY:
    print("Gemini API key detected. GenAI explanations are enabled with REST calls.")
else:
    print("WARNING: GEMINI_API_KEY not found. GenAI explanations will be disabled.")
# ---------------------------------------------

# 4. Define the input data schema using Pydantic
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

# 5. Standard prediction endpoint (Your original code)
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    if len(transaction.features) != len(feature_names):
        raise HTTPException(status_code=400, detail=f"Exactly {len(feature_names)} features required.")
    
    input_data = pd.DataFrame([transaction.features], columns=feature_names)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": float(probability),
        "status": "Transaction Blocked" if prediction == 1 else "Transaction Approved"
    }

# --- 6. NEW: Step 9 Endpoint with GenAI Copilot ---
@app.post("/predict_and_explain")
def predict_and_explain(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    input_data = pd.DataFrame([transaction.features], columns=feature_names)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Base response
    response = {
        "is_fraud": bool(prediction),
        "fraud_probability": float(probability),
        "status": "Transaction Blocked" if prediction == 1 else "Transaction Approved",
        "explanation": "Transaction normal. No anomalies detected.",
        "genai_status": "not-applicable",
        "genai_prompt": None
    }

    # If it's fraud, trigger the LLM to explain it!
    if prediction == 1:
        response["genai_prompt"] = _build_fraud_prompt(probability)
        try:
            response["explanation"] = _generate_fraud_explanation(probability)
            response["genai_status"] = "generated"
        except RuntimeError as exc:
            response["explanation"] = _fallback_explanation(probability)
            response["genai_status"] = "fallback"
            if EXPOSE_GENAI_ERRORS:
                response["genai_error"] = str(exc)
            
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)