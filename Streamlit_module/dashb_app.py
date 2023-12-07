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
client_ids = input_data.index.tolist()
feature_means = joblib.load("Streamlit_module/feature_means.joblib")
explainer = joblib.load("Streamlit_module/local_explainer.joblib")
list_features = feature_means.index.tolist()
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

def create_histogram(client_id, data, feature, data_mean_features):
    client_id = int(client_id)
    client_row = data.loc[[client_id]].drop('SK_ID_CURR', axis=1, errors='ignore')
    client_value = client_row[feature].values[0]
    # Crea l'istogramma
    fig, ax = plt.subplots()
    categories = ['Client Value', 'Good Clients Mean', 'Bad Clients Mean']
    values = [client_value, data_mean_features['Good Clients'].values[0], data_mean_features['Bad Clients'].values[0]]
    colors = ['skyblue', 'green', 'red']

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(values)), values, color=colors)  # Sostituisci i colori come desideri


    ax.bar(categories, values, color=colors)
    ax.set_title(f'Feature: {selected_feature}')
    ax.set_ylabel('Value')
    ax.set_ylim([min(values) - abs(min(values)) * 0.1, max(values) + abs(max(values)) * 0.1])  # Aggiunge spazio sopra e sotto le barre
    for i, v in enumerate(values):
            ax.text(i, v + (max(values) - min(values)) * 0.05, f'{v:.2f}', color=colors[i], ha='center')

    # Mostra il grafico
    st.pyplot(fig)


st.title("Select a feature from the list, compare the selected client with bad and good clients means")
selected_feature = str(st.selectbox("Select feature", list_features))
if st.button('choose feature'):
    create_histogram(client_id=selected_client_id, data=input_data, feature=selected_feature, data_mean_features= feature_means)
    st.write(feature_means.loc[selected_feature])


st.title("Global importance of features")

# explainer = shap.TreeExplainer(lightgbmodel)

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



    # streamlit run .\Streamlit\dashb_app.py


