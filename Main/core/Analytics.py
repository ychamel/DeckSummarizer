import uuid
import pymongo
import streamlit as st



# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(**st.secrets["mongo"])

@st.cache_resource
def get_id():
    return uuid.uuid4()

client = init_connection()

def set_data(query, answer):
    try:
        client.Streamlit.insert_one({"name": str(get_id()), "question": query, "answer": answer})
    except Exception as e:
        print(f"failed to connect to server withe error: {e}")

