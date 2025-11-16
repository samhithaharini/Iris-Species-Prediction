import streamlit as st
import numpy as np
import joblib
model = joblib.load("iris_logreg.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Iris Classification Using Logistic Regression")
st.write("Enter the flower measurements:")
sl = st.number_input("Sepal Length (SL)", 0.0, 10.0)
sw = st.number_input("Sepal Width (SW)", 0.0, 10.0)
pl = st.number_input("Petal Length (PL)", 0.0, 10.0)
pw = st.number_input("Petal Width (PW)", 0.0, 10.0)
if st.button("Predict"):
    features = np.array([[sl, sw, pl, pw]])
    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    st.success(f"Predicted CLASS: **{pred}**")