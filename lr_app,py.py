import streamlit as  st
import pandas as pd
import numpy as np
import sklearn
import pickle

model = pickle.load(open(r"linear_regression_model.pkl",'rb'))

# LETS CREATE WEB APP

st.title("Scikit-Learn Linear Regression model")
tv = st.text_input("Enter TV sales...")
radio = st.text_input("Enter Radio sales...")
newspaper = st.text_input("Enter Newspaper sales...")


if st.button("Predict"):
    features = np.array([[tv,radio,newspaper]], dtype=np.float64)
    result = model.predict(features).reshape(1,-1)
    st.write("Predicted sale::::", result[0])

