import streamlit as st
import requests
import pandas as pd
import joblib
import shap_functions as shap_f
import matplotlib.pyplot as plt
import shap
from streamlit_shap import st_shap
import numpy as np


input_data = joblib.load("Streamlit_module/data4streamlit_light.joblib")
#input_data = joblib.load("Streamlit_module/sample_ten_rows.joblib")
lightgbmodel =  joblib.load("Streamlit_module/lightgbmodelsh.joblib")
feature_means = joblib.load("Streamlit_module/feature_means.joblib")
feature_means_all = joblib.load("Streamlit_module/feature_means_all.joblib")
data4histo = joblib.load("Streamlit_module/data4histogram.joblib")
client_ids = data4histo.index.tolist()  # input_data.index.tolist()
list_features = feature_means_all.index.tolist()
# explainer = joblib.load("Streamlit_module/local_explainer.joblib")

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

    # Controlla la decisione e stampa il messaggio corrispondente
    if decision == 'accepted':
        # Stampa la decisione in verde con un checkmark
        st.markdown(f"<h2 style='color:green;'>✔️ The loan request for client {selected_client_id} is {decision}</h2>", unsafe_allow_html=True)
    elif decision == 'refused':
        # Stampa la decisione in rosso con una x
        st.markdown(f"<h2 style='color:red;'>❌ The loan request for client {selected_client_id} is {decision}</h2>", unsafe_allow_html=True)

st.title("Global importance of features")

explainer = shap.TreeExplainer(lightgbmodel)

shap_values = explainer.shap_values(input_data.drop('SK_ID_CURR', axis=1))
# Creazione di una figura in Matplotlib
plt.figure(figsize=(10, 5))  # Puoi regolare le dimensioni a seconda delle tue esigenze

# Generazione del summary plot
shap.summary_plot(shap_values, input_data.drop('SK_ID_CURR', axis=1), show=False)

# Visualizzazione del plot in Streamlit
st.pyplot(plt)
# shap.summary_plot(shap_values, input_data.drop('SK_ID_CURR', axis=1))

st.title("Importance by customer")

shap_f.generate_force_plot(selected_client_id, input_data, explainer)

shap_f.generate_waterfall_plot(selected_client_id, input_data, explainer)

st.title("Select a feature from the list, compare the selected client with bad and good clients means")
selected_feature = str(st.selectbox("Select feature", list_features))
if st.button('choose feature'):
    shap_f.create_histogram(client_id=selected_client_id, data=data4histo, feature=selected_feature, data_mean_features= feature_means_all)
    # print the 2 columns that will be plotted, useful for debugging
    st.write(feature_means_all.loc[selected_feature])


    # streamlit run .\Streamlit\dashb_app.py


