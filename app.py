from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load clean model components
obj = joblib.load("svm_clean.pkl")
scaler = obj["scaler"]
svm = obj["svm"]
classes = obj["classes"]

app = FastAPI(
    title="Fatigue Prediction API",
    description="Predict worker fatigue level using clean SVM model",
    version="1.0"
)

class WorkerData(BaseModel):
    sleep_hours: float
    shift_hours: float
    opf_minutes: float
    ppe_violations: int
    high_risk_events: int
    break_compliance: float
    movement_score: float

@app.get("/")
def home():
    return {"status": "Fatigue API is running!"}

@app.post("/predict")
def predict(data: WorkerData):

    # Convert to numpy array
    X = np.array([[
        data.sleep_hours,
        data.shift_hours,
        data.opf_minutes,
        data.ppe_violations,
        data.high_risk_events,
        data.break_compliance,
        data.movement_score
    ]])

    # Scale numeric features
    X_scaled = scaler.transform(X)

    # Probability predictions
    probs = svm.predict_proba(X_scaled)[0]

    # Most likely class
    pred_class = classes[np.argmax(probs)]

    return {
        "fatigue_label": pred_class,
        "probabilities": {cls: float(p) for cls, p in zip(classes, probs)}
    }
