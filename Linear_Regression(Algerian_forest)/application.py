import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the Ridge regression model and StandardScaler from pickles
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

def predict_datapoint(temperature, rh, ws, rain, ffmc, dmc, isi, classes, region):
    new_data_scaled = standard_scaler.transform([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
    result = ridge_model.predict(new_data_scaled)
    return result[0]

def main():
    st.title("Wildfire Prediction")

    # Sidebar with input fields
    st.sidebar.header("Input Parameters")
    temperature = st.sidebar.slider("Temperature", 0.0, 40.0, 25.0)
    rh = st.sidebar.slider("Relative Humidity", 0.0, 100.0, 50.0)
    ws = st.sidebar.slider("Wind Speed", 0.0, 10.0, 5.0)
    rain = st.sidebar.slider("Rain", 0.0, 25.0, 0.0)
    ffmc = st.sidebar.slider("FFMC", 50.0, 100.0, 75.0)
    dmc = st.sidebar.slider("DMC", 0.0, 300.0, 150.0)
    isi = st.sidebar.slider("ISI", 0.0, 20.0, 10.0)
    classes = st.sidebar.slider("Fire Classes", 0.0, 5.0, 2.0)
    region = st.sidebar.slider("Region", 1.0, 10.0, 5.0)

    # Prediction
    if st.button("Predict"):
        result = predict_datapoint(temperature, rh, ws, rain, ffmc, dmc, isi, classes, region)
        st.success(f"The predicted result is: {result}")

if __name__ == "__main__":
    main()
