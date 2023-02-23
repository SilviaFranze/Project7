# 1. Library imports
import joblib
import uvicorn
from fastapi import FastAPI
from credit_def_risk import Train
import numpy as np
import pickle
import pandas as pd
import json
import requests


# 2. Create the app object
app = FastAPI()
scaler = joblib.load("../standardscaler.joblib")
light_classif = joblib.load("../lightgbmodel.joblib")
features = joblib.load("../features.joblib")
input_data = joblib.load("../input_data.joblib")

@app.get('/')
@app.get('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application
    """
    return {'message': 'system up'}

#   Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the prediction on the client with the probability
@app.get('/prediction')

def get_client(x_test):
    features_list=[]
    values_list=[]
    for k,v in x_test.sample().to_dict().items():
        features_list.append(k)
        for key,val in v.items():
            values_list.append(val)

    return dict(zip(features_list, values_list))

def predict(data:Train):
    data_dict = data.dict()
    data_df = pd.DataFrame.from_dict([data_dict])
    data_df = data_df[features]

    data_df = scaler.transform(data_df)
    print(data_df, flush = True)

    prediction = light_classif.predict(data_df)
    print(prediction, flush=True)

    # threshold determined after studying the costs for each threshold
    if prediction <0.90:
        prediction_label = "Credit refused"
    if prediction >=0.90:
        prediction_label = "Credit accepted"

    return {"prediction":prediction_label}

def get_client(x_test):
    features_list=[]
    values_list=[]
    for k,v in x_test.sample().to_dict().items():
        features_list.append(k)
        for key,val in v.items():
            values_list.append(val)

    return dict(zip(features_list, values_list))

response = requests.get("http://127.0.0.1:8000//predict", json=get_client(input_data))
print(response.text)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)

#uvicorn app:app --reload
