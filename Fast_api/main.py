# 1. Library imports
import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from lightgbm import LGBMClassifier

app = FastAPI()

#Load customer data
input_data_scaled = joblib.load("../Data&output/X_tst_sld_skid.joblib")
# Load the LightGBM model
lgbm_classif = joblib.load("../Data&output/lightgbmodel.joblib")



@app.get('/prediction/{id_client}')


def prediction_client(id_client: int):   
    '''
    Endpoint to get the client id and return the prediction based on a pre trained LightGBM model
    '''    

    # Select customer data specified by ID and dropping the ID column
    selected_customer = input_data_scaled[input_data_scaled['SK_ID_CURR'] == id_client].drop('SK_ID_CURR', axis=1)

    # makes the prediction on the index given as input
    predizione = lgbm_classif.predict_proba(selected_customer)[:,0][0]
    
    # determines whether the application was accepted or rejected on the basis of the 0.90 threshold
    if predizione > 0.90:
        decision = "accepted"
    else:
        decision = "refused"

    # returns a dictionary with the client ID and the decision made
    return {"Customer id": id_client, 
            "Decision": decision}


if __name__ == '__main__':
	uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn main:app --reload



"""
a voir ou rajouter ce partie la dans le get de la prediction

	client = input_data_scaled.loc[input_data_scaled['SK_ID_CURR'] == {id_client}].iloc[:,1:]
	pd.DataFrame(light_classif.predict_proba(client))
"""