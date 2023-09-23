from flask import Flask, request, jsonify, abort
import joblib
import git
import os
import hmac
import hashlib
import json


secret_token = os.getenv("SECRET_TOKEN")

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

#Load customer data
input_data_scaled = joblib.load("/home/silviafranze/input_data_scaled.joblib")   #   X_tst_sld_skid.joblib was the original one
# Load the LightGBM model
lgbm_classif = joblib.load("/home/silviafranze/lightgbmodel.joblib")

app = Flask(__name__)
app.config["DEBUG"] = True


def is_valid_signature(x_hub_signature, data, private_key):
    # x_hub_signature and data are from the webhook payload
    # private key is your webhook secret
    hash_algorithm, github_signature = x_hub_signature.split('=', 1)
    algorithm = hashlib.__dict__.get(hash_algorithm)
    encoded_key = bytes(private_key, 'latin-1')
    mac = hmac.new(encoded_key, msg=data, digestmod=algorithm)
    return hmac.compare_digest(mac.hexdigest(), github_signature)


@app.route('/')
def home():
    return 'Welcome to the Homepage'


@app.route('/update_fromgithub', methods=['POST'])
def webhook():

    if request.method != 'POST':
        return 'OK'
    else:
        abort_code = 418
        # Do initial validations on required headers
        if 'X-Github-Event' not in request.headers:
            abort(abort_code)
        if 'X-Github-Delivery' not in request.headers:
            abort(abort_code)
        if 'X-Hub-Signature' not in request.headers:
            abort(abort_code)
        if not request.is_json:
            abort(abort_code)
        if 'User-Agent' not in request.headers:
            abort(abort_code)
        ua = request.headers.get('User-Agent')
        if not ua.startswith('GitHub-Hookshot/'):
            abort(abort_code)

        event = request.headers.get('X-GitHub-Event')
        if event == "ping":
            return json.dumps({'msg': 'Hi!'})
        if event != "push":
            return json.dumps({'msg': "Wrong event type"})

        x_hub_signature = request.headers.get('X-Hub-Signature')
        # webhook content type should be application/json for request.data to have the payload
        # request.data is empty in case of x-www-form-urlencoded
        if not is_valid_signature(x_hub_signature, request.data, secret_token):
            print('Deploy signature failed: {sig}'.format(sig=x_hub_signature))
            abort(abort_code)

        payload = request.get_json()
        if payload is None:
            print('Deploy payload is empty: {payload}'.format(
                payload=payload))
            abort(abort_code)

        if payload['ref'] != 'refs/heads/frompythonanywhere':
            return json.dumps({'msg': 'Not frompythonanywhere branch; ignoring'})

        repo = git.Repo("/home/silviafranze/Project7")
        repo.git.checkout('frompythonanywhere') # Checkout to branch "frompythonanywhere"
        origin = repo.remotes.origin

        pull_info = origin.pull()

        if len(pull_info) == 0:
            return json.dumps({'msg': "Didn't pull any information from remote!"})
        if pull_info[0].flags > 128:
            return json.dumps({'msg': "Didn't pull any information from remote!"})

        commit_hash = pull_info[0].commit.hexsha
        build_commit = f'build_commit = "{commit_hash}"'
        print(f'{build_commit}')
        return 'Updated PythonAnywhere server to commit {commit}'.format(commit=commit_hash)



    '''
    if request.method == 'POST':
        repo = git.Repo("/home/silviafranze/Project7")
        repo.git.checkout('frompythonanywhere') # Checkout to branch "frompythonanywhere"
        origin = repo.remotes.origin
        origin.pull()
        return 'Updated PythonAnywhere successfully', 200
    else:
        return 'Wrong event type', 400
    '''



@app.route('/prediction/<int:id_client>', methods =['GET'])
def prediction(id_client):

    '''
    Endpoint to get the client id and return the prediction based on a pre trained LightGBM model
    '''

    # Select customer data specified by ID
    # selected_customer = input_data_scaled.loc[id_client].to_numpy().reshape(1,-1)
    selected_customer = input_data_scaled.loc[id_client].values.reshape(1,-1)


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


    # return f'This will be the prediction score app!{id_client}'
   # testing webhook3