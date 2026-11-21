
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Obesity Level Predictor")

model = joblib.load("/app/artifacts/model.joblib")
le = joblib.load("/app/artifacts/label_encoder.joblib")
feature_columns = joblib.load("/app/artifacts/feature_columns.joblib")

class ObesityInput(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

@app.post("/predict")
def predict(data: ObesityInput):
    # Превращаем вход в DataFrame с теми же колонками, что при обучении
    row = pd.DataFrame([data.dict()])
    row = pd.get_dummies(row, drop_first=True)
    # Выравниваем колонки по обучению
    row = row.reindex(columns=feature_columns, fill_value=0)
    pred = model.predict(row)[0]
    label = le.inverse_transform([pred])[0]
    return {"prediction": label}

@app.get("/health")
def health():
    return {"status": "ok"}
