import requests

BASE_URL = "http://silviafranze.pythonanywhere.com"

def test_valid_client_id():
    client_id = 105548  # Example valid client ID that exists in your dataset
    response = requests.get(f"{BASE_URL}/prediction/{client_id}")
    
    
    # Check if the response status code is 200 (OK)
    assert response.status_code == 200

def test_invalid_client_id_format():
    client_id = 207122  # Example invalid client ID format
    response = requests.get(f"{BASE_URL}/prediction/{client_id}")
    
    # Check if the response status code is 500, the one I set
    assert response.status_code == 500


""" def test_invalid_client_id_not_in_dataset():
    client_id = 999999  # Example valid format but not in your dataset
    response = requests.get(f"{BASE_URL}/prediction/{client_id}")
    
    # Check if the response status code is 500 (Internal Server Error)
    assert response.status_code == 500
    # Check the error message
    assert response.json()["error"] == "An internal error occurred. Please try again later."
 """