# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("turnover_api")

# Create input/output pydantic models
input_model = create_model("turnover_api_input", **{'Age': 29, 'BusinessTravel': 2, 'DailyRate': 408, 'Department': 1, 'DistanceFromHome': 25, 'Education': 5, 'EducationField': 5, 'EmployeeCount': 1, 'EnvironmentSatisfaction': 3, 'Gender': 1, 'HourlyRate': 71, 'JobInvolvement': 2, 'JobLevel': 1, 'JobRole': 6, 'JobSatisfaction': 2, 'MaritalStatus': 1, 'MonthlyIncome': 2546, 'MonthlyRate': 18300, 'NumCompaniesWorked': 5, 'Over18': 1, 'OverTime': 0, 'PercentSalaryHike': 16, 'PerformanceRating': 3, 'RelationshipSatisfaction': 2, 'StandardHours': 80, 'StockOptionLevel': 0, 'TotalWorkingYears': 6, 'TrainingTimesLastYear': 2, 'WorkLifeBalance': 4, 'YearsAtCompany': 2, 'YearsInCurrentRole': 2, 'YearsSinceLastPromotion': 1, 'YearsWithCurrManager': 1})
output_model = create_model("turnover_api_output", prediction=1)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
