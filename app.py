import streamlit as st
import pickle
import requests

@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/manan36chauhan/data_science_project/First-init/adult_income_model.pkl"
    response = requests.get(url)
    model = pickle.loads(response.content)
    return model

model = load_model()
