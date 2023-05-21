
# A very simple Flask Hello World app for you to get started with...
from flask import Flask
import joblib
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

app = Flask(__name__)

# Autentication Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Autentication through local browser
drive = GoogleDrive(gauth)



##Load customer data
#input_data_scaled = joblib.load("/home/silviafranze/X_tst_sld_skid.joblib")
## Load the LightGBM model
#lgbm_classif = joblib.load("/home/silviafranze/lightgbmodel.joblib")




#  files ID Google Drive
input_data_scld_id = "1evpnU161MKAKQ1xUrDP4OOi2G0uAhpTu"
lgbm_classif_id = "1VtW9HyNbUID8thQJQvmo7z4-b-3L3QfB"

# Download  files from Google Drive
input_data_scld = drive.CreateFile({'id': input_data_scld_id})
input_data_scld.GetContentFile('X_tst_sld_skid.joblib')

lgbm_model = drive.CreateFile({'id': lgbm_classif_id})
lgbm_model.GetContentFile('lightgbmodel.joblib')



# model and data load
input_data_scaled = joblib.load("input_data.joblib")
lgbm_classif = joblib.load("model.joblib")



@app.route('/ciao')
def hello_pythonanywhere():
    return 'Hello from PythonAnywhere!'

if __name__ == '__main__':
    app.run()