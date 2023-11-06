import streamlit as st
import requests
import pandas as pd
import joblib
from Streamlit_module import shap_functions as shap_f
import lightgbm
import shap
from streamlit_shap import st_shap

import os
st.write("CIAoooooooooooo, Current directory:", os.getcwd())
st.write("Files in current directory:", os.listdir('.'))

input_data = joblib.load("Streamlit/input_data_str_light.joblib")
lightgbmodel =  joblib.load("Streamlit/lightgbmodelsh.joblib")
client_ids = input_data.SK_ID_CURR.tolist()

# add the line to generate the explainer
# add the line that insulates the client ids to make the liste deroulante

st.title("Scoring prediction")
st.write('Select the customer\'s ID to make a prediction on their loan request.')


# URL FastAPI
api_url = "https://silviafranze.pythonanywhere.com/prediction"

# Client id selection through a list
st.subheader("Client selection:")
selected_client_id = str(st.selectbox("Select client ID", client_ids))

# button to make the choice from the list of client ids to make the  GET request to the API
if st.button('Predict'):
    # Effettua la richiesta GET passando l'id del cliente come parametro
    response = requests.get(api_url + "/" + selected_client_id)

    # Retrieves the decision from the API respnse 
    decision = response.json()['Decision']

    # writes the final decision
    st.write('The decision for the client num', selected_client_id, 'is', decision)


st.title("Global importance of features")

shap.initjs()  # JavaScript plots


explainer = shap.TreeExplainer(lightgbmodel, model_output='raw')



    # streamlit run .\Streamlit\dashb_app.py
