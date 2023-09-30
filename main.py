from flask import Flask, request, jsonify, abort
import joblib
import git
import os
import hmac
import hashlib
import json
import logging

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

logging.basicConfig(level=logging.DEBUG)

def is_valid_signature(x_hub_signature, data, private_key):
    """
    Validate the X-Hub-Signature header value against the request payload.

    Args:
    - x_hub_signature (str): The value of X-Hub-Signature header, e.g., "sha1=abcd1234".
    - data (bytes): The raw request payload.
    - private_key (str): The webhook secret token.

    Returns:
    - bool: True if the signature is valid; False otherwise.
    """
    # Ensure data is in bytes format
    if not isinstance(data, bytes):
        logging.error("Data is not in bytes format.")
        return False

    # Split the header value into hash algorithm and signature
    try:
        hash_algorithm, github_signature = x_hub_signature.split('=', 1)
    except ValueError:
        logging.error("X-Hub-Signature header value is not in the expected format.")
        return False

    # Get the appropriate hash function from hashlib
    algorithm = hashlib.__dict__.get(hash_algorithm)
    if not algorithm:
        logging.error(f"Hash algorithm {hash_algorithm} is not supported.")
        return False

    # Encode the private key into bytes
    encoded_key = bytes(private_key, 'latin-1')

    # Compute the HMAC
    mac = hmac.new(encoded_key, msg=data, digestmod=algorithm)
    computed_signature = mac.hexdigest()

    # Log the computed signature for debugging purposes (optional)
    logging.debug(f"Computed HMAC signature: {computed_signature}")

    # Compare the computed signature with the one provided by GitHub
    return hmac.compare_digest(computed_signature, github_signature)


@app.route('/')
def home():
    return 'Welcome to the Homepage'

@app.route('/update_fromgithub', methods=['POST'])
def webhook():

    if request.method != 'POST':
        return 'OK'
    else:
        # Do initial validations on required headers
        if 'X-Github-Event' not in request.headers:
            print("Missing X-Github-Event header")
            abort(400, description="Missing X-Github-Event header")
        if 'X-Github-Delivery' not in request.headers:
            print("Missing X-Github-Delivery header")
            abort(400, description="Missing X-Github-Delivery header")
        if 'X-Hub-Signature' not in request.headers:
            print("Missing X-Hub-Signature header")
            abort(400, description="Missing X-Hub-Signature header")
        if not request.is_json:
            print("Request is not JSON")
            abort(400, description="Request is not JSON")
        if 'User-Agent' not in request.headers:
            print("Missing User-Agent header")
            abort(400, description="Missing User-Agent header")

        ua = request.headers.get('User-Agent')  # Moved this line out of the block
        print("User-Agent:", ua)

        if not ua.startswith('GitHub-Hookshot/'):
            print("Invalid User-Agent header")
            abort(400, description="Invalid User-Agent header")
        event = request.headers.get('X-GitHub-Event')
        print("X-GitHub-Event:", event)
        if event == "ping":
            return json.dumps({'msg': 'Hi!'})
        if event != "push":
            return json.dumps({'msg': "Wrong event type"})

        # Get both signatures from the headers
        x_hub_signature = request.headers.get('X-Hub-Signature')
        x_hub_signature_256 = request.headers.get('X-Hub-Signature-256')

        # Define the abort code
        abort_code = 418

        # Check the sha1 signature
        if x_hub_signature and not is_valid_signature(x_hub_signature, request.data, secret_token):
            print('Deploy sha1 signature failed: {sig}'.format(sig=x_hub_signature))
            abort(abort_code)

        # Check the sha256 signature
        if x_hub_signature_256 and not is_valid_signature(x_hub_signature_256, request.data, secret_token):
            print('Deploy sha256 signature failed: {sig}'.format(sig=x_hub_signature_256))
            abort(abort_code)

        # If neither signature is provided, abort
        if not x_hub_signature and not x_hub_signature_256:
            print('Missing X-Hub-Signature and X-Hub-Signature-256 headers')
            abort(abort_code)


        payload = request.get_json()
        if payload is None:
            print('Deploy payload is empty: {payload}'.format(
                payload=payload))
            abort(abort_code)

        print(payload['ref']+'CIAO AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        if payload['ref'] != 'refs/heads/master':
            return json.dumps({'msg': 'Not master; ignoring'})

        repo = git.Repo("/home/silviafranze/Project7")
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

@app.errorhandler(Exception)
def handle_generic_exception(e):
    # Log the exception for debugging purposes
    app.logger.error(f"An error occurred: {str(e)}")
    
    # Return a generic error message to the client
    return jsonify(error="An internal error occurred. Please try again later."), 500

@app.route('/prediction/<int:id_client>', methods =['GET'])
def prediction(id_client):

    # If any exception occurs here, it will be caught by the generic handler

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

 # testing  webhook