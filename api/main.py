from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Credit Approval API", version="0.1.0")

# Load artifacts at startup
model = joblib.load("models/credit_model.pkl")
scaler = joblib.load("models/scaler.pkl")

class Applicant(BaseModel):
    age: int = Field(ge=18, le=100)
    income: float = Field(ge=0)
    credit_score: int = Field(ge=300, le=850)
    loan_amount: float = Field(gt=0)

@app.post("/predict")
def predict(applicant: Applicant):
    data = np.array([[applicant.age, applicant.income, applicant.credit_score, applicant.loan_amount]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)
    return {
        "approved": bool(pred[0]),
        "note": "Educational demo only. Not for real credit decisions."
    }
