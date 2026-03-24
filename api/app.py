from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model and features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# Risk tier logic
def get_tier(score):
    if score < 25:
        return "Low Risk"
    elif score < 50:
        return "Medium Risk"
    elif score < 75:
        return "High Risk"
    else:
        return "Critical Risk"

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # Ensure all required features exist
    for col in features:
        if col not in df:
            df[col] = 0

    df = df[features]

    prob = model.predict_proba(df)[0][1]
    score = round(prob * 100, 1)
    tier = get_tier(score)

    return {
        "churn_probability": round(prob, 4),
        "risk_score": score,
        "risk_tier": tier
    }