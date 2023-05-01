# 1. Library imports
import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from lightgbm import LGBMClassifier

app = FastAPI()
input_data_scaled = joblib.load("../Data&output/test_scaled_data_ok.joblib")
light_classif = joblib.load("../Data&output/lightgbmodel.joblib")


# utiliser le get pour choisir l id

@app.get('/prediction/{id_client}')


def predizione_lightgbm(numero):
    # loads the trained LightGBM model
    model = lgb.Booster(model_file='modello_lgb.txt')
    
    # makes the prediction on the index given as input
    predizione = model.predict([numero])[0]
    
    # determines whether the application was accepted or rejected on the basis of the 0.90 threshold
    if predizione > 0.90:
        decision = "accepted"
    else:
        decision = "refused"


def read_id(id_client: int):
	"""
	Endpoint to	"""

	return {'message': id_client,
			'decision': 'accordé'}    #


if __name__ == '__main__':
	uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn main:app --reload



"""
a voir ou rajouter ce partie la dans le get de la prediction

	client = input_data_scaled.loc[input_data_scaled['SK_ID_CURR'] == {id_client}].iloc[:,1:]
	pd.DataFrame(light_classif.predict_proba(client))
"""