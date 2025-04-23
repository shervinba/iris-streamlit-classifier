import streamlit as st
import numpy as np
from joblib import load

# Load the trained model
model = load("models/model.pkl")

# Page title
st.title("🌸 Iris Flower Classifier")

# Input sliders
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

# Prepare features array
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prediction
if st.button("Predict"):
    pred = model.predict(features)[0]
    classes = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"🌼 Prediction: {classes[pred]}")
