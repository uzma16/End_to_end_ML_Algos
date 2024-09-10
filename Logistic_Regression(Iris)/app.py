import streamlit as st
import pandas as pd
import pickle

# Load the model
model = pickle.load(open('classification.pkl', 'rb'))

# Streamlit App
st.title("Binary Classification with Iris Dataset")

# Sidebar with user input
st.sidebar.header("User Input:")
sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)


# Display prediction
st.write("Prediction:")
 # Prediction
if st.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if prediction[0] == 1:
        result = "Versicolour"
    else:
        result = "Setosa"    
    st.success(f"The predicted result is: {result}")