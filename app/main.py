from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is Running"}

@app.post("/predict")
def predict(data: CustomerData):

    row = {feature: 0 for feature in feature_names}

    row["SeniorCitizen"] = data.SeniorCitizen
    row["tenure"] = data.tenure
    row["MonthlyCharges"] = data.MonthlyCharges
    row["TotalCharges"] = data.TotalCharges

    input_df = pd.DataFrame([row])

    prediction = model.predict(input_df)

    return {
        "prediction": int(prediction[0]),
        "result": "Will Churn" if prediction[0] == 1 else "Will Not Churn"
    }