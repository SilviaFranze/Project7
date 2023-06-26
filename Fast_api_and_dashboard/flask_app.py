
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

#Load customer data
input_data_scaled = joblib.load("/home/silviafranze/X_tst_sld_skid.joblib")     
# Load the LightGBM model
lgbm_classif = joblib.load("/home/silviafranze/lightgbmodel.joblib")

@app.route('/prediction/<int:id_client>', methods=['GET'])



def prediction(id_client):   
    '''
    Endpoint to get the client id and return the prediction based on a pre trained LightGBM model
    '''    

    # Select customer data specified by ID and drops the ID column
    selected_customer = input_data_scaled[input_data_scaled['SK_ID_CURR'] == id_client].drop('SK_ID_CURR', axis=1)

    # makes the prediction about a specific client
    prediction = lgbm_classif.predict_proba(selected_customer)[:,0][0]
    
    # determines whether the application was accepted or rejected on the basis of the 0.90 threshold
    if prediction > 0.90:
        decision = "accepted"
    else:
        decision = "refused"

    # returns a dictionary with the client ID and the decision made
    response = {"Customer id": id_client, 
                "Decision": decision}
    
    return jsonify(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0")