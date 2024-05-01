# Importing librairies
import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

# Inserting title
st.title('Employee Attrition')

# Loading the model
@st.cache_resource
def model_loaded():
    return load_model('turnover_api')

model = model_loaded()

# Setting the screen
col1, col2 = st.columns([1,3])

# Inputs (features)

age = col1.text_input('Age:', '29', max_chars=5)
businesstravel = col1.text_input('BusinessTravel:', '2', max_chars=5)
dailyrate = col1.text_input('DailyRate:', '408', max_chars=5)
department = col1.text_input('Department:', '1', max_chars=5)
distancefromhome = col1.text_input('DistanceFromHome:', '25', max_chars=5)
education = col1.text_input('Education:', '5', max_chars=5)
educationfield = col1.text_input('EducationField:', '5', max_chars=5)
employeecount = col1.text_input('EmployeeCount:', '1', max_chars=5)
environmentsatisfaction = col1.text_input('EnvironmentSatisfaction:', '3', max_chars=5)
gender = col1.text_input('Gender:', '1', max_chars=5)
hourlyrate = col1.text_input('HourlyRate:', '71', max_chars=5)
jobinvolvement = col1.text_input('JobInvolvement:', '2', max_chars=5)
joblevel = col1.text_input('JobLevel:', '1', max_chars=5)
jobrole = col1.text_input('JobRole:', '6', max_chars=5)
jobsatisfaction = col1.text_input('JobSatisfaction:', '2', max_chars=5)
maritalstatus = col1.text_input('MaritalStatus:', '1', max_chars=5)
monthlyincome = col1.text_input('MonthlyIncome:', '2546', max_chars=5)
monthlyrate = col1.text_input('MonthlyRate:', '18300', max_chars=5)
numcompaniesworked = col1.text_input('NumCompaniesWorked:', '5', max_chars=5)
over18 = col1.text_input('Over18:', '1', max_chars=5)
overtime = col1.text_input('OverTime:', '0', max_chars=5)
percentsalaryhike = col1.text_input('PercentSalaryHike:', '16', max_chars=5)
performancerating = col1.text_input('PerformanceRating:', '3', max_chars=5)
relationshipsatisfaction = col1.text_input('RelationshipSatisfaction:', '2', max_chars=5)
standardhours = col1.text_input('StandardHours:', '80', max_chars=5)
stockoptionlevel = col1.text_input('StockOptionLevel:', '0', max_chars=5)
totalworkingyears = col1.text_input('TotalWorkingYears:', '6', max_chars=5)
trainingtimeslastyear = col1.text_input('TrainingTimesLastYear:', '2', max_chars=5)
worklifebalance = col1.text_input('WorkLifeBalance:', '4', max_chars=5)
yearsatcompany = col1.text_input('YearsAtCompany:', '2', max_chars=5)
yearsincurrentrole = col1.text_input('YearsInCurrentRole:', '2', max_chars=5)
yearssincelastpromotion = col1.text_input('YearsSinceLastPromotion:', '1', max_chars=5)
yearswithcurrmanager = col1.text_input('YearsWithCurrManager:', '1', max_chars=5)

# transtyping
def try_parse(str_value):
    try:
        value = float(str_value)
    except Exception:
        value = float('NaN')
    return value

# converting into dictionary
my_data = {
    'age': try_parse(age),
    'businesstravel': try_parse(businesstravel),
    'dailyrate': try_parse(dailyrate),
    'department': try_parse(department),
    'distancefromhome': try_parse(distancefromhome),
    'education': try_parse(education),
    'educationfield': try_parse(educationfield),
    'employeecount': try_parse(employeecount),
    'environmentsatisfaction': try_parse(environmentsatisfaction),
    'gender': try_parse(gender),
    'hourlyrate': try_parse(hourlyrate),
    'jobinvolvement': try_parse(jobinvolvement),
    'joblevel': try_parse(joblevel),
    'jobrole': try_parse(jobrole),
    'jobsatisfaction': try_parse(jobsatisfaction),
    'maritalstatus': try_parse(maritalstatus),
    'monthlyincome': try_parse(monthlyincome),
    'monthlyrate': try_parse(monthlyrate),
    'numcompaniesworked': try_parse(numcompaniesworked),
    'over18': try_parse(over18),
    'overtime': try_parse(overtime),
    'percentsalaryhike': try_parse(percentsalaryhike),
    'performancerating': try_parse(performancerating),
    'relationshipsatisfaction': try_parse(relationshipsatisfaction),
    'standardhours': try_parse(standardhours),
    'stockoptionlevel': try_parse(stockoptionlevel),
    'totalworkingyears': try_parse(totalworkingyears),
    'trainingtimeslastyear': try_parse(trainingtimeslastyear),
    'worklifebalance': try_parse(worklifebalance),
    'yearsatcompany': try_parse(yearsatcompany),
    'yearsincurrentrole': try_parse(yearsincurrentrole),
    'yearssincelastpromotion': try_parse(yearssincelastpromotion),
    'yearswithcurrmanager': try_parse(yearswithcurrmanager)
}

# predicting the attrition class
try:
    predicted_value = predict_model(model, data = pd.DataFrame([my_data]))
except Exception:
    predicted_value = 'unknown'

# adding calculation button
st.button('Calculate')

# printing the results
st.write('Attrition class:', predicted_value['prediction_label'].iloc[0],
         predicted_value['prediction_score'].iloc[0])