import streamlit as st
import pandas as pd
import pickle
import requests


@st.cache_resource
def load_model():
    url = "https://github.com/manan36chauhan/data_science_project/raw/First-init/adult_income_model.pkl"
    response = requests.get(url)
    return pickle.loads(response.content)

model = load_model()

st.title("Adult Income Prediction App")


age = st.slider("Age", 18, 90)
education_num = st.slider("Education Level (numeric)", 1, 16)
hours_per_week = st.slider("Hours Worked Per Week", 1, 100)
gender = st.selectbox("Gender", ["Male", "Female"])


gender_encoded = 1 if gender == "Male" else 0


input_data = pd.DataFrame({
    'age': [age],
    'education-num': [education_num],
    'hours-per-week': [hours_per_week],
    'gender': [gender_encoded]
})


if st.button("Predict Income Level"):
    prediction = model.predict(input_data)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income: {result}")
