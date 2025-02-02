from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib 
import pandas as pd
from loan_status import LoanStatus

model = joblib.load('loan_status_predictor.pkl')

app = FastAPI()

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To FastAPI': f'{name}'}

@app.post("/predict")
async def predict_loan_status(data: LoanStatus):
    #data = data.model_dump()
    data = pd.DataFrame([data.model_dump()])
    result = model.predict(data)

    if result[0] == 1:
        return {'Loan Status': "Approved"}
    else:
        return {'Loan Status': "Not Approved"}

