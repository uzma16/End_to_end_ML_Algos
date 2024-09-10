import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the StandardScaler
scaler = pickle.load(open('models/StandardScaler.pkl', 'rb'))

# Load the prediction model
model = pickle.load(open('models/prediction_model.pkl', 'rb'))

# Streamlit App
st.title("Diabetes Prediction App")

# Sidebar with user input
st.sidebar.header("User Input:")
pregnancies = st.sidebar.slider("Number of Pregnancies", 0, 17, 3)
glucose = st.sidebar.slider("Glucose Level", 0, 199, 117)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 72)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 23)
insulin = st.sidebar.slider("Insulin Level", 0, 846, 30)
bmi = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.3725)
age = st.sidebar.slider("Age", 21, 81, 29)

# Create a DataFrame with user input
user_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# Scale the user input using the loaded StandardScaler
scaled_input = scaler.transform(user_data)

# Make the prediction
prediction = model.predict(scaled_input)

# Display prediction
st.subheader("Prediction:")
if st.button("Predict"):
    if prediction[0] == 1:
        st.write("High likelihood of diabetes.")
    else:
        st.write("Low likelihood of diabetes.")

# Show user input data
st.subheader("User Input Data:")
st.write(user_data)
