# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib 
# import pandas as pd
# import uvicorn
# from loan_status import LoanStatus
# import pickle
# import numpy as np
# app = FastAPI()

# pickle_in = open("loan_status_predictor.pkl","rb")
# model = pickle.load(pickle_in)

# @app.get('/{name}')
# def get_name(name: str):
#     return {'Welcome To FastAPI': f'{name}'}



# @app.post("/predict/")
# async def predict_loan_status(data: LoanStatus):
#     data = data.model_dump()
#     Gender = data['Gender']
#     Married =  data['Married']
#     Dependents = data['Dependents']
#     Education = data['Education']
#     Self_Employed = data['Self_Employed']
#     ApplicantIncome = data['ApplicantIncome']
#     CoapplicantIncome = data['CoapplicantIncome']
#     LoanAmount = data['LoanAmount']
#     Loan_Amount_Term =  data['Loan_Amount_Term']
#     Credit_History = data['Credit_History']
#     Property_Area = data['Property_Area']

#     # Predict loan status
#     prediction = model.predict(data)
#     if prediction [0] == 1:
#         return {'Loan Status': "Approved"}
#     else:
#         return {'Loan Status': "Not Approved"}

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib 
import pandas as pd
from loan_status import LoanStatus

model = joblib.load('loan_status_predictor.pkl')

app = FastAPI()

@app.post("/predict")
async def predict_loan_status(data: LoanStatus):
    #data = data.model_dump()
    data = pd.DataFrame([data.model_dump()])
    result = model.predict(data)

    if result[0] == 1:
        return {'Loan Status': "Approved"}
    else:
        return {'Loan Status': "Not Approved"}

