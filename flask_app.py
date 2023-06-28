
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/')
def hello_world():
    return 'Hello from my shiny Flask app!'

@app.route('/prediction/<int:id_client>', methods=['GET'])
def prediction(id_client):
    return id_client
