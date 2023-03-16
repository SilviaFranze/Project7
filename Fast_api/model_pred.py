from pydantic import BaseModel
import json
import requests
import joblib
import pandas as pd

input_data = joblib.load("../Data&output/input_data.joblib")
input_data_scaled = joblib.load("../Data&output/input_data_scaled.joblib")
scaler = joblib.load("../Data&output/standardscaler.joblib")
light_classif = joblib.load("../Data&output/lightgbmodel.joblib")
features = joblib.load("../Data&output/features.joblib")


class Credit_Model():
 # 6. Class constructor, loads the model
 def __init__(self):
  self.lgbmodel = joblib.load("../Data&output/lightgbmodel.joblib")
  self.input_data = joblib.load("../Data&output/input_data_scaled.joblib")
  self.data_in = joblib.load("../Data&output/features.joblib")
 def get_entry(self):
  """
  :param x_test: test  dataset from which we can take one entry (new client)
  :return: json with param:value pair
  """
  features_list=[]
  values_list=[]
  for k,v in self.input_data.sample().to_dict().items():
   features_list.append(k)
   for key,val in v.items():
    values_list.append(val)

  return json.dumps(dict(zip(features_list, values_list)))

# notes session 24 fevrier
# endpoint qui demande ID client qui demande fichier test scaled
 # creer un dataframe pour le client passe
 #je garde les info du dataframe (sousdataframe)
 # user donne un id user qui va etre use pour filtre le datframe
 # apres je fais le predict avec la ligne
 #



 # input_dict = Model.get_entry(input_data)    (?)


 def pred_credit (self, CODE_GENDER ,
                  FLAG_OWN_CAR ,
                  FLAG_OWN_REALTY ,
                  REGION_POPULATION_RELATIVE ,
                  DAYS_REGISTRATION ,
                  DAYS_ID_PUBLISH ,
                  FLAG_WORK_PHONE ,
                  FLAG_PHONE ,
                  REGION_RATING_CLIENT ,
                  REG_REGION_NOT_LIVE_REGION ,
                  REG_REGION_NOT_WORK_REGION ,
                  REG_CITY_NOT_LIVE_CITY ,
                  REG_CITY_NOT_WORK_CITY ,
                  EXT_SOURCE_1 ,
                  EXT_SOURCE_2 ,
                  EXT_SOURCE_3 ,
                  BASEMENTAREA_MODE ,
                  ELEVATORS_MODE ,
                  LANDAREA_MODE ,
                  OBS_60_CNT_SOCIAL_CIRCLE ,
                  DEF_60_CNT_SOCIAL_CIRCLE ,
                  FLAG_DOCUMENT_2 ,
                  AMT_REQ_CREDIT_BUREAU_MON ,
                  AMT_REQ_CREDIT_BUREAU_QRT ,
                  AMT_REQ_CREDIT_BUREAU_YEAR ,
                  NAME_CONTRACT_TYPE_Cashloans ,
                  NAME_CONTRACT_TYPE_Revolvingloans ,
                  NAME_INCOME_TYPE_Commercialassociate ,
                  NAME_INCOME_TYPE_Maternityleave ,
                  NAME_INCOME_TYPE_Pensioner ,
                  NAME_INCOME_TYPE_Stateservant ,
                  NAME_INCOME_TYPE_Unemployed ,
                  NAME_INCOME_TYPE_Working ,
                  NAME_EDUCATION_TYPE_Academicdegree ,
                  NAME_EDUCATION_TYPE_Highereducation ,
                  NAME_EDUCATION_TYPE_Lowersecondary ,
                  NAME_EDUCATION_TYPE_Secondary_secondaryspecial ,
                  NAME_FAMILY_STATUS_Civilmarriage ,
                  NAME_FAMILY_STATUS_Married ,
                  NAME_FAMILY_STATUS_Single_notmarried ,
                  NAME_FAMILY_STATUS_Widow ,
                  NAME_HOUSING_TYPE_House_apartment ,
                  NAME_HOUSING_TYPE_Officeapartment ,
                  NAME_HOUSING_TYPE_Rentedapartment ,
                  NAME_HOUSING_TYPE_Withparents ,
                  OCCUPATION_TYPE_Accountants ,
                  OCCUPATION_TYPE_Cleaningstaff ,
                  OCCUPATION_TYPE_Cookingstaff ,
                  OCCUPATION_TYPE_Corestaff ,
                  OCCUPATION_TYPE_Drivers ,
                  OCCUPATION_TYPE_Highskilltechstaff ,
                  OCCUPATION_TYPE_Low_skillLaborers ,
                  OCCUPATION_TYPE_Managers ,
                  OCCUPATION_TYPE_Medicinestaff ,
                  OCCUPATION_TYPE_Privateservicestaff ,
                  OCCUPATION_TYPE_Salesstaff ,
                  OCCUPATION_TYPE_Securitystaff ,
                  OCCUPATION_TYPE_Waiters_barmenstaff ,
                  WEEKDAY_APPR_PROCESS_START_MONDAY ,
                  WEEKDAY_APPR_PROCESS_START_TUESDAY ,
                  ORGANIZATION_TYPE_Agriculture ,
                  ORGANIZATION_TYPE_Bank ,
                  ORGANIZATION_TYPE_BusinessEntityType3 ,
                  ORGANIZATION_TYPE_Construction ,
                  ORGANIZATION_TYPE_Government ,
                  ORGANIZATION_TYPE_Industry_type1 ,
                  ORGANIZATION_TYPE_Industry_type12 ,
                  ORGANIZATION_TYPE_Industry_type3 ,
                  ORGANIZATION_TYPE_Industry_type4 ,
                  ORGANIZATION_TYPE_Industry_type9 ,
                  ORGANIZATION_TYPE_Kindergarten ,
                  ORGANIZATION_TYPE_Medicine ,
                  ORGANIZATION_TYPE_Military ,
                  ORGANIZATION_TYPE_Police ,
                  ORGANIZATION_TYPE_Restaurant ,
                  ORGANIZATION_TYPE_School ,
                  ORGANIZATION_TYPE_Security ,
                  ORGANIZATION_TYPE_SecurityMinistries ,
                  ORGANIZATION_TYPE_Self_employed ,
                  ORGANIZATION_TYPE_Trade_type3 ,
                  ORGANIZATION_TYPE_Trade_type6 ,
                  ORGANIZATION_TYPE_Trade_type7 ,
                  ORGANIZATION_TYPE_Transport_type3 ,
                  ORGANIZATION_TYPE_Transport_type4 ,
                  ORGANIZATION_TYPE_University ,
                  ORGANIZATION_TYPE_XNA ,
                  DAYS_EMPLOYED_PERC ,
                  INCOME_PER_PERSON ,
                  ANNUITY_INCOME_PERC ,
                  PAYMENT_RATE ):
  data_in = [[CODE_GENDER ,
              FLAG_OWN_CAR ,
              FLAG_OWN_REALTY ,
              REGION_POPULATION_RELATIVE ,
              DAYS_REGISTRATION ,
              DAYS_ID_PUBLISH ,
              FLAG_WORK_PHONE ,
              FLAG_PHONE ,
              REGION_RATING_CLIENT ,
              REG_REGION_NOT_LIVE_REGION ,
              REG_REGION_NOT_WORK_REGION ,
              REG_CITY_NOT_LIVE_CITY ,
              REG_CITY_NOT_WORK_CITY ,
              EXT_SOURCE_1 ,
              EXT_SOURCE_2 ,
              EXT_SOURCE_3 ,
              BASEMENTAREA_MODE ,
              ELEVATORS_MODE ,
              LANDAREA_MODE ,
              OBS_60_CNT_SOCIAL_CIRCLE ,
              DEF_60_CNT_SOCIAL_CIRCLE ,
              FLAG_DOCUMENT_2 ,
              AMT_REQ_CREDIT_BUREAU_MON ,
              AMT_REQ_CREDIT_BUREAU_QRT ,
              AMT_REQ_CREDIT_BUREAU_YEAR ,
              NAME_CONTRACT_TYPE_Cashloans ,
              NAME_CONTRACT_TYPE_Revolvingloans ,
              NAME_INCOME_TYPE_Commercialassociate ,
              NAME_INCOME_TYPE_Maternityleave ,
              NAME_INCOME_TYPE_Pensioner ,
              NAME_INCOME_TYPE_Stateservant ,
              NAME_INCOME_TYPE_Unemployed ,
              NAME_INCOME_TYPE_Working ,
              NAME_EDUCATION_TYPE_Academicdegree ,
              NAME_EDUCATION_TYPE_Highereducation ,
              NAME_EDUCATION_TYPE_Lowersecondary ,
              NAME_EDUCATION_TYPE_Secondary_secondaryspecial ,
              NAME_FAMILY_STATUS_Civilmarriage ,
              NAME_FAMILY_STATUS_Married ,
              NAME_FAMILY_STATUS_Single_notmarried ,
              NAME_FAMILY_STATUS_Widow ,
              NAME_HOUSING_TYPE_House_apartment ,
              NAME_HOUSING_TYPE_Officeapartment ,
              NAME_HOUSING_TYPE_Rentedapartment ,
              NAME_HOUSING_TYPE_Withparents ,
              OCCUPATION_TYPE_Accountants ,
              OCCUPATION_TYPE_Cleaningstaff ,
              OCCUPATION_TYPE_Cookingstaff ,
              OCCUPATION_TYPE_Corestaff ,
              OCCUPATION_TYPE_Drivers ,
              OCCUPATION_TYPE_Highskilltechstaff ,
              OCCUPATION_TYPE_Low_skillLaborers ,
              OCCUPATION_TYPE_Managers ,
              OCCUPATION_TYPE_Medicinestaff ,
              OCCUPATION_TYPE_Privateservicestaff ,
              OCCUPATION_TYPE_Salesstaff ,
              OCCUPATION_TYPE_Securitystaff ,
              OCCUPATION_TYPE_Waiters_barmenstaff ,
              WEEKDAY_APPR_PROCESS_START_MONDAY ,
              WEEKDAY_APPR_PROCESS_START_TUESDAY ,
              ORGANIZATION_TYPE_Agriculture ,
              ORGANIZATION_TYPE_Bank ,
              ORGANIZATION_TYPE_BusinessEntityType3 ,
              ORGANIZATION_TYPE_Construction ,
              ORGANIZATION_TYPE_Government ,
              ORGANIZATION_TYPE_Industry_type1 ,
              ORGANIZATION_TYPE_Industry_type12 ,
              ORGANIZATION_TYPE_Industry_type3 ,
              ORGANIZATION_TYPE_Industry_type4 ,
              ORGANIZATION_TYPE_Industry_type9 ,
              ORGANIZATION_TYPE_Kindergarten ,
              ORGANIZATION_TYPE_Medicine ,
              ORGANIZATION_TYPE_Military ,
              ORGANIZATION_TYPE_Police ,
              ORGANIZATION_TYPE_Restaurant ,
              ORGANIZATION_TYPE_School ,
              ORGANIZATION_TYPE_Security ,
              ORGANIZATION_TYPE_SecurityMinistries ,
              ORGANIZATION_TYPE_Self_employed ,
              ORGANIZATION_TYPE_Trade_type3 ,
              ORGANIZATION_TYPE_Trade_type6 ,
              ORGANIZATION_TYPE_Trade_type7 ,
              ORGANIZATION_TYPE_Transport_type3 ,
              ORGANIZATION_TYPE_Transport_type4 ,
              ORGANIZATION_TYPE_University ,
              ORGANIZATION_TYPE_XNA ,
              DAYS_EMPLOYED_PERC ,
              INCOME_PER_PERSON ,
              ANNUITY_INCOME_PERC ,
              PAYMENT_RATE ]]
  prediction = light_classif.predict(data_in)
  probability = light_classif.predict_proba(data_in).max()
  return prediction[0], probability




