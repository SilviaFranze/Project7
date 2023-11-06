import streamlit as st
import requests
import pandas as pd
import joblib
import shap_functions as shap_f
import lightgbm
import shap
from streamlit_shap import st_shap


input_data = joblib.load("Streamlit_module/input_data_str_light.joblib")
lightgbmodel =  joblib.load("Streamlit_module/lightgbmodelsh.joblib")
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

explainer = shap.TreeExplainer(lightgbmodel, model_output='raw')

shap_values = explainer.shap_values(input_data.drop('SK_ID_CURR', axis=1))
shap.summary_plot(shap_values, input_data.drop('SK_ID_CURR', axis=1))

st.title("Importance by customer")

shap_f.generate_force_plot(selected_client_id, input_data.drop('SK_ID_CURR', axis=1), explainer)

shap_f.generate_waterfall_plot(selected_client_id, input_data, explainer)

    # streamlit run .\Streamlit\dashb_app.py
