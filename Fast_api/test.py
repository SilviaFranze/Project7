from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib

features = joblib.load("../Data&output/features.joblib")

class Item(BaseModel):
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


app = FastAPI()


@app.post("/items/")
async def create_item(item: Item):
	# print("ciao")
	return item.name

if __name__ == '__main__':
	uvicorn.run(app, host='127.0.0.1', port=8000)
