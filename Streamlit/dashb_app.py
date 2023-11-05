import streamlit as st
import requests
import streamlit as st
from streamlit_shap import st_shap
import pandas as pd
import shap
import joblib
import io

import os
st.write("CIAoooooooooooo, Current directory:", os.getcwd())
st.write("Files in current directory:", os.listdir('.'))

#input_data = joblib.load("project7/Streamlit/data4streamlit.joblib")     # /home/silviafranze pour le run sur python anywhere  # substitute them with the actual functioning dataset, to calculate the explainer etc
#client_ids =  joblib.load("/mount/src/project7/Streamlit/list_id_clients_long.joblib")
# URL del file raw su GitHub
url = "https://github.com/SilviaFranze/Project7/raw/master/Streamlit/list_id_clients_long.joblib"

# Effettua una richiesta GET per scaricare il file
response = requests.get(url)

# Verifica che la richiesta sia andata a buon fine
response.raise_for_status()

# Crea un oggetto file-like dai dati raw
file_like_object = io.BytesIO(response.content)

# Carica i dati joblib dall'oggetto file-like
input_data = joblib.load(file_like_object)


"""
# Token di Accesso Personale (recuperato dai Secrets di Streamlit)
pat = st.secrets["DB_USERNAME"]
pat2 = st.secrets["DB_TOKEN"]

# Imposta l'autenticazione per la richiesta
headers = {'Authorization': f'token {pat}'}

# Scarica i dati raw dai file su GitHub
response_data = requests.get(url_data, headers=headers)
response_data.raise_for_status()  # Verifica che la richiesta sia andata a buon fine

response_ids = requests.get(url_ids, headers=headers)
response_ids.raise_for_status()  # Verifica che la richiesta sia andata a buon fine

# Crea oggetti file-like dai dati raw
file_like_object_data = io.BytesIO(response_data.content)
file_like_object_ids = io.BytesIO(response_ids.content)

# Carica i dati joblib dagli oggetti file-like
input_data = joblib.load(file_like_object_data)
client_ids = joblib.load(file_like_object_ids)

"""

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



    # streamlit run .\Streamlit\dashb_app.py
