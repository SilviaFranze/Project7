from pydantic import BaseModel
import json
import requests
import joblib

input_data = joblib.load("../Data&output/input_data.joblib")
input_data_scaled = joblib.load("../Data&output/input_data_scaled.joblib")
scaler = joblib.load("../Data&output/standardscaler.joblib")
light_classif = joblib.load("../Data&output/lightgbmodel.joblib")


def get_entry(x_test):
    features_list=[]
    values_list=[]
    for k,v in x_test.sample().to_dict().items():
        features_list.append(k)
        for key,val in v.items():
            values_list.append(val)

    return json.dumps(dict(zip(features_list, values_list)))


input_dict = get_entry(input_data)


def predict_proba(**dict_entry):
    data_in = dict_entry
    prediction = light_classif.predict(data_in)
    probability = light_classif.predict_proba(data_in)
    return prediction[0], probability




class Train(BaseModel):
    """
    Class which describes Train variables dataset
    """
    CODE_GENDER: int
    FLAG_OWN_CAR: int
    FLAG_OWN_REALTY: int
    REGION_POPULATION_RELATIVE: float
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: int
    FLAG_WORK_PHONE: int
    FLAG_PHONE: int
    REGION_RATING_CLIENT: int
    REG_REGION_NOT_LIVE_REGION: int
    REG_REGION_NOT_WORK_REGION: int
    REG_CITY_NOT_LIVE_CITY: int
    REG_CITY_NOT_WORK_CITY: int
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    BASEMENTAREA_MODE: float
    ELEVATORS_MODE: float
    LANDAREA_MODE: float
    OBS_60_CNT_SOCIAL_CIRCLE: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    FLAG_DOCUMENT_2: int
    AMT_REQ_CREDIT_BUREAU_MON: float
    AMT_REQ_CREDIT_BUREAU_QRT: float
    AMT_REQ_CREDIT_BUREAU_YEAR: float
    NAME_CONTRACT_TYPE_Cashloans: int
    NAME_CONTRACT_TYPE_Revolvingloans: int
    NAME_INCOME_TYPE_Commercialassociate: int
    NAME_INCOME_TYPE_Maternityleave: int
    NAME_INCOME_TYPE_Pensioner: int
    NAME_INCOME_TYPE_Stateservant: int
    NAME_INCOME_TYPE_Unemployed: int
    NAME_INCOME_TYPE_Working: int
    NAME_EDUCATION_TYPE_Academicdegree: int
    NAME_EDUCATION_TYPE_Highereducation: int
    NAME_EDUCATION_TYPE_Lowersecondary: int
    NAME_EDUCATION_TYPE_Secondary_secondaryspecial: int
    NAME_FAMILY_STATUS_Civilmarriage: int
    NAME_FAMILY_STATUS_Married: int
    NAME_FAMILY_STATUS_Single_notmarried: int
    NAME_FAMILY_STATUS_Widow: int
    NAME_HOUSING_TYPE_House_apartment: int
    NAME_HOUSING_TYPE_Officeapartment: int
    NAME_HOUSING_TYPE_Rentedapartment: int
    NAME_HOUSING_TYPE_Withparents: int
    OCCUPATION_TYPE_Accountants: int
    OCCUPATION_TYPE_Cleaningstaff: int
    OCCUPATION_TYPE_Cookingstaff: int
    OCCUPATION_TYPE_Corestaff: int
    OCCUPATION_TYPE_Drivers: int
    OCCUPATION_TYPE_Highskilltechstaff: int
    OCCUPATION_TYPE_Low_skillLaborers: int
    OCCUPATION_TYPE_Managers: int
    OCCUPATION_TYPE_Medicinestaff: int
    OCCUPATION_TYPE_Privateservicestaff: int
    OCCUPATION_TYPE_Salesstaff: int
    OCCUPATION_TYPE_Securitystaff: int
    OCCUPATION_TYPE_Waiters_barmenstaff: int
    WEEKDAY_APPR_PROCESS_START_MONDAY: int
    WEEKDAY_APPR_PROCESS_START_TUESDAY: int
    ORGANIZATION_TYPE_Agriculture: int
    ORGANIZATION_TYPE_Bank: int
    ORGANIZATION_TYPE_BusinessEntityType3: int
    ORGANIZATION_TYPE_Construction: int
    ORGANIZATION_TYPE_Government: int
    ORGANIZATION_TYPE_Industry_type1: int
    ORGANIZATION_TYPE_Industry_type12: int
    ORGANIZATION_TYPE_Industry_type3: int
    ORGANIZATION_TYPE_Industry_type4: int
    ORGANIZATION_TYPE_Industry_type9: int
    ORGANIZATION_TYPE_Kindergarten: int
    ORGANIZATION_TYPE_Medicine: int
    ORGANIZATION_TYPE_Military: int
    ORGANIZATION_TYPE_Police: int
    ORGANIZATION_TYPE_Restaurant: int
    ORGANIZATION_TYPE_School: int
    ORGANIZATION_TYPE_Security: int
    ORGANIZATION_TYPE_SecurityMinistries: int
    ORGANIZATION_TYPE_Self_employed: int
    ORGANIZATION_TYPE_Trade_type3: int
    ORGANIZATION_TYPE_Trade_type6: int
    ORGANIZATION_TYPE_Trade_type7: int
    ORGANIZATION_TYPE_Transport_type3: int
    ORGANIZATION_TYPE_Transport_type4: int
    ORGANIZATION_TYPE_University: int
    ORGANIZATION_TYPE_XNA: int
    DAYS_EMPLOYED_PERC: float
    INCOME_PER_PERSON: float
    ANNUITY_INCOME_PERC: float
    PAYMENT_RATE: float