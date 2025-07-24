import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model
model = pickle.load(open("models/house_price_model.pkl", "rb"))

st.set_page_config(page_title=" House Price Predictor", layout="centered")
st.title(" House Price Prediction App")

st.markdown("Enter the house details below to predict the price (in $1000s):")

# Feature inputs
CRIM = st.number_input("CRIM (per capita crime rate)", min_value=0.0, value=0.1)
ZN = st.number_input("ZN (residential land zoned)", min_value=0.0, value=0.0)
INDUS = st.number_input("INDUS (non-retail business area)", min_value=0.0, value=5.0)
CHAS = st.selectbox("CHAS (Charles River dummy variable)", [0, 1])
NOX = st.number_input("NOX (nitric oxides concentration)", min_value=0.0, value=0.5)
RM = st.number_input("RM (average number of rooms)", min_value=1.0, value=6.0)
AGE = st.number_input("AGE (% built before 1940)", min_value=0.0, value=30.0)
DIS = st.number_input("DIS (distance to employment centres)", min_value=0.0, value=5.0)
RAD = st.number_input("RAD (access to radial highways)", min_value=1.0, value=1.0)
TAX = st.number_input("TAX (property-tax rate)", min_value=100.0, value=300.0)
PTRATIO = st.number_input("PTRATIO (pupil-teacher ratio)", min_value=10.0, value=15.0)
B = st.number_input("B (blacks per town)", min_value=0.0, value=390.0)
LSTAT = st.number_input("LSTAT (% lower status population)", min_value=0.0, value=12.0)

if st.button("Predict Price"):
    features = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    price = model.predict(features)[0]
    st.success(f"💰 Predicted House Price: ${price * 1000:,.2f}")
