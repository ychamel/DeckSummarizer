import json
import requests


def parse_img(name, byte_img):
    # Define the URL and endpoint of the API you want to connect to
    api_url = "https://api.doxter.ai/"
    endpoint = "analyzer/run/"

    # Define the necessary headers and authentication for the request
    headers = {
        "Authorization": "Token ac560906e09a3b9c501a64f6e68888151fdd4ae4"
    }
    # file_name = str(uuid.uuid4())
    file = {"image": (name, byte_img, "image/jpeg")}
    data = {"project": 1063, "run_document_detector": True, "run_table_detector": False, "rtl": True}
    # Send a POST request to create a new customer
    response = requests.post(api_url + endpoint, headers=headers, data=data, files=file)

    # Check the response status code to ensure your request was successful
    if not response.ok:
        print("Error: {}".format(response.content))
        return None

    return json.loads(response.content)

def fetch_text(uuid):
    # Define the URL and endpoint of the API you want to connect to
    api_url = "https://api.doxter.ai/"
    endpoint = f"analyzer/result/{uuid}"

    # Define the necessary headers and authentication for the request
    headers = {
        "Authorization": "Token ac560906e09a3b9c501a64f6e68888151fdd4ae4"
    }

    # Send a POST request to create a new customer
    response = requests.get(api_url + endpoint, headers=headers)

    # Check the response status code to ensure your request was successful
    if not response.ok:
        print("Error: {}".format(response.content))
        return None

    return json.loads(response.content)