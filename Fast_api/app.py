# 1. Library imports
import joblib
import uvicorn
from fastapi import FastAPI
from model_pred import Credit_Model
from pydantic import BaseModel
# from lightgbm import LGBMClassifier
# import numpy as np
# import pickle
# import pandas as pd
# import json
# import requests

input_data_scaled = joblib.load("../Data&output/input_data_scaled.joblib")
light_classif = joblib.load("../Data&output/lightgbmodel.joblib")

class Train(BaseModel):
    """
	Class which describes Train variables dataset
	"""
    CODE_GENDER: float
    FLAG_OWN_CAR: float
    FLAG_OWN_REALTY: float
    REGION_POPULATION_RELATIVE: float
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: float
    FLAG_WORK_PHONE: float
    FLAG_PHONE: float
    REGION_RATING_CLIENT: float
    REG_REGION_NOT_LIVE_REGION: float
    REG_REGION_NOT_WORK_REGION: float
    REG_CITY_NOT_LIVE_CITY: float
    REG_CITY_NOT_WORK_CITY: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    BASEMENTAREA_MODE: float
    ELEVATORS_MODE: float
    LANDAREA_MODE: float
    OBS_60_CNT_SOCIAL_CIRCLE: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    FLAG_DOCUMENT_2: float
    AMT_REQ_CREDIT_BUREAU_MON: float
    AMT_REQ_CREDIT_BUREAU_QRT: float
    AMT_REQ_CREDIT_BUREAU_YEAR: float
    NAME_CONTRACT_TYPE_Cashloans: float
    NAME_CONTRACT_TYPE_Revolvingloans: float
    NAME_INCOME_TYPE_Commercialassociate: float
    NAME_INCOME_TYPE_Maternityleave: float
    NAME_INCOME_TYPE_Pensioner: float
    NAME_INCOME_TYPE_Stateservant: float
    NAME_INCOME_TYPE_Unemployed: float
    NAME_INCOME_TYPE_Working: float
    NAME_EDUCATION_TYPE_Academicdegree: float
    NAME_EDUCATION_TYPE_Highereducation: float
    NAME_EDUCATION_TYPE_Lowersecondary: float
    NAME_EDUCATION_TYPE_Secondary_secondaryspecial: float
    NAME_FAMILY_STATUS_Civilmarriage: float
    NAME_FAMILY_STATUS_Married: float
    NAME_FAMILY_STATUS_Single_notmarried: float
    NAME_FAMILY_STATUS_Widow: float
    NAME_HOUSING_TYPE_House_apartment: float
    NAME_HOUSING_TYPE_Officeapartment: float
    NAME_HOUSING_TYPE_Rentedapartment: float
    NAME_HOUSING_TYPE_Withparents: float
    OCCUPATION_TYPE_Accountants: float
    OCCUPATION_TYPE_Cleaningstaff: float
    OCCUPATION_TYPE_Cookingstaff: float
    OCCUPATION_TYPE_Corestaff: float
    OCCUPATION_TYPE_Drivers: float
    OCCUPATION_TYPE_Highskilltechstaff: float
    OCCUPATION_TYPE_Low_skillLaborers: float
    OCCUPATION_TYPE_Managers: float
    OCCUPATION_TYPE_Medicinestaff: float
    OCCUPATION_TYPE_Privateservicestaff: float
    OCCUPATION_TYPE_Salesstaff: float
    OCCUPATION_TYPE_Securitystaff: float
    OCCUPATION_TYPE_Waiters_barmenstaff: float
    WEEKDAY_APPR_PROCESS_START_MONDAY: float
    WEEKDAY_APPR_PROCESS_START_TUESDAY: float
    ORGANIZATION_TYPE_Agriculture: float
    ORGANIZATION_TYPE_Bank: float
    ORGANIZATION_TYPE_BusinessEntityType3: float
    ORGANIZATION_TYPE_Construction: float
    ORGANIZATION_TYPE_Government: float
    ORGANIZATION_TYPE_Industry_type1: float
    ORGANIZATION_TYPE_Industry_type12: float
    ORGANIZATION_TYPE_Industry_type3: float
    ORGANIZATION_TYPE_Industry_type4: float
    ORGANIZATION_TYPE_Industry_type9: float
    ORGANIZATION_TYPE_Kindergarten: float
    ORGANIZATION_TYPE_Medicine: float
    ORGANIZATION_TYPE_Military: float
    ORGANIZATION_TYPE_Police: float
    ORGANIZATION_TYPE_Restaurant: float
    ORGANIZATION_TYPE_School: float
    ORGANIZATION_TYPE_Security: float
    ORGANIZATION_TYPE_SecurityMinistries: float
    ORGANIZATION_TYPE_Self_employed: float
    ORGANIZATION_TYPE_Trade_type3: float
    ORGANIZATION_TYPE_Trade_type6: float
    ORGANIZATION_TYPE_Trade_type7: float
    ORGANIZATION_TYPE_Transport_type3: float
    ORGANIZATION_TYPE_Transport_type4: float
    ORGANIZATION_TYPE_University: float
    ORGANIZATION_TYPE_XNA: float
    DAYS_EMPLOYED_PERC: float
    INCOME_PER_PERSON: float
    ANNUITY_INCOME_PERC: float
    PAYMENT_RATE: float


# 2. Create the app object
app = FastAPI()
model = Credit_Model()
features = joblib.load("../Data&output/features.joblib")

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
# @app.get('/{name}')
# def get_name(name: str):
#     return {'message': f'Hello, {name}'}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence

@app.post('/predict')    # dans mon dataset de test il faut avoir l ID du client
                        # filtre le df pour prendre clui de l ID
                        # predict_proba de ce partie la
                       # dans la dashboard je donne une liste deroulante avec le differentes id que l user va choisir
                     # la dashboard va appeler mon API
                    #je peut faire le GET direcrement et je mets le parametre de l ID en entre
                #streamlit je vais mettre un selectioner, il va faire un appel a l api, il paut tourner en local

def pred_probability(data: Train):
    data = data.dict()
    prediction, probability = model.pred_credit(
        data["CODE_GENDER"],
        data["FLAG_OWN_CAR"],data["FLAG_OWN_REALTY"],
        data["REGION_POPULATION_RELATIVE"],
        data["DAYS_REGISTRATION"],
        data["DAYS_ID_PUBLISH"],
        data["FLAG_WORK_PHONE"],
        data["FLAG_PHONE"],
        data["REGION_RATING_CLIENT"],
        data["REG_REGION_NOT_LIVE_REGION"],
        data["REG_REGION_NOT_WORK_REGION"],
        data["REG_CITY_NOT_LIVE_CITY"],
        data["REG_CITY_NOT_WORK_CITY"],
        data["EXT_SOURCE_1"],
        data["EXT_SOURCE_2"],
        data["EXT_SOURCE_3"],
        data["BASEMENTAREA_MODE"],
        data["ELEVATORS_MODE"],
        data["LANDAREA_MODE"],
        data["OBS_60_CNT_SOCIAL_CIRCLE"],
        data["DEF_60_CNT_SOCIAL_CIRCLE"],
        data["FLAG_DOCUMENT_2"],
        data["AMT_REQ_CREDIT_BUREAU_MON"],
        data["AMT_REQ_CREDIT_BUREAU_QRT"],
        data["AMT_REQ_CREDIT_BUREAU_YEAR"],
        data["NAME_CONTRACT_TYPE_Cashloans"],
        data["NAME_CONTRACT_TYPE_Revolvingloans"],
        data["NAME_INCOME_TYPE_Commercialassociate"],
        data["NAME_INCOME_TYPE_Maternityleave"],
        data["NAME_INCOME_TYPE_Pensioner"],
        data["NAME_INCOME_TYPE_Stateservant"],
        data["NAME_INCOME_TYPE_Unemployed"],
        data["NAME_INCOME_TYPE_Working"],
        data["NAME_EDUCATION_TYPE_Academicdegree"],
        data["NAME_EDUCATION_TYPE_Highereducation"],
        data["NAME_EDUCATION_TYPE_Lowersecondary"],
        data["NAME_EDUCATION_TYPE_Secondary_secondaryspecial"],
        data["NAME_FAMILY_STATUS_Civilmarriage"],
        data["NAME_FAMILY_STATUS_Married"],
        data["NAME_FAMILY_STATUS_Single_notmarried"],
        data["NAME_FAMILY_STATUS_Widow"],
        data["NAME_HOUSING_TYPE_House_apartment"],
        data["NAME_HOUSING_TYPE_Officeapartment"],
        data["NAME_HOUSING_TYPE_Rentedapartment"],
        data["NAME_HOUSING_TYPE_Withparents"],
        data["OCCUPATION_TYPE_Accountants"],
        data["OCCUPATION_TYPE_Cleaningstaff"],
        data["OCCUPATION_TYPE_Cookingstaff"],
        data["OCCUPATION_TYPE_Corestaff"],
        data["OCCUPATION_TYPE_Drivers"],
        data["OCCUPATION_TYPE_Highskilltechstaff"],
        data["OCCUPATION_TYPE_Low_skillLaborers"],
        data["OCCUPATION_TYPE_Managers"],
        data["OCCUPATION_TYPE_Medicinestaff"],
        data["OCCUPATION_TYPE_Privateservicestaff"],
        data["OCCUPATION_TYPE_Salesstaff"],
        data["OCCUPATION_TYPE_Securitystaff"],
        data["OCCUPATION_TYPE_Waiters_barmenstaff"],
        data["WEEKDAY_APPR_PROCESS_START_MONDAY"],
        data["WEEKDAY_APPR_PROCESS_START_TUESDAY"],
        data["ORGANIZATION_TYPE_Agriculture"],
        data["ORGANIZATION_TYPE_Bank"],
        data["ORGANIZATION_TYPE_BusinessEntityType3"],
        data["ORGANIZATION_TYPE_Construction"],
        data["ORGANIZATION_TYPE_Government"],
        data["ORGANIZATION_TYPE_Industry_type1"],
        data["ORGANIZATION_TYPE_Industry_type12"],
        data["ORGANIZATION_TYPE_Industry_type3"],
        data["ORGANIZATION_TYPE_Industry_type4"],
        data["ORGANIZATION_TYPE_Industry_type9"],
        data["ORGANIZATION_TYPE_Kindergarten"],
        data["ORGANIZATION_TYPE_Medicine"],
        data["ORGANIZATION_TYPE_Military"],
        data["ORGANIZATION_TYPE_Police"],
        data["ORGANIZATION_TYPE_Restaurant"],
        data["ORGANIZATION_TYPE_School"],
        data["ORGANIZATION_TYPE_Security"],
        data["ORGANIZATION_TYPE_SecurityMinistries"],
        data["ORGANIZATION_TYPE_Self_employed"],
        data["ORGANIZATION_TYPE_Trade_type3"],
        data["ORGANIZATION_TYPE_Trade_type6"],
        data["ORGANIZATION_TYPE_Trade_type7"],
        data["ORGANIZATION_TYPE_Transport_type3"],
        data["ORGANIZATION_TYPE_Transport_type4"],
        data["ORGANIZATION_TYPE_University"],
        data["ORGANIZATION_TYPE_XNA"],
        data["DAYS_EMPLOYED_PERC"],
        data["INCOME_PER_PERSON"],
        data["ANNUITY_INCOME_PERC"],
        data["PAYMENT_RATE"])    # [data[i].sample() for i in features]

    return {
        'prediction': prediction,
        'probability': probability
    }



# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn app:app --reload

#%%
