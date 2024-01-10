import datetime
import uuid

import requests
import json

import streamlit as st


@st.cache_data
def get_id():
    return str(uuid.uuid4())


def set_data(query, answer):
    url = "https://eu-west-2.aws.data.mongodb-api.com/app/data-iruhp/endpoint/data/v1/action/findOne"

    payload = json.dumps({
        "collection": "Synapse",
        "database": "DataAnalytics",
        "dataSource": "Cluster0",
        "document": {
            "id": get_id(),
            "question": query,
            "answer": answer,
            "time": str(datetime.datetime.now())
        }
    })
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Request-Headers': '*',
        'api-key': st.session_state.get("MONGODB_API_KEY"),
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response)