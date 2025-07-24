import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ----- Custom CSS -----
custom_css = """
<style>
body {
    background-color: #0f172a;
    color: #f1f5f9;
}
h1, h2, h3 {
    color: #38bdf8;
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 5px;
    padding: 8px 16px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---- Title ----
st.title("🏡 House Price Prediction (13 Features)")
st.markdown("Enter the house details below:")

# ---- Input Fields ----
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq. ft):", min_value=300, max_value=10000, value=1500)
    bedrooms = st.number_input("Bedrooms:", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms:", min_value=1, max_value=10, value=2)
    stories = st.number_input("Stories:", min_value=1, max_value=4, value=2)
    parking = st.number_input("Parking Spaces:", min_value=0, max_value=5, value=1)
    house_age = st.number_input("House Age (in years):", min_value=0, max_value=100, value=5)
    furnished = st.selectbox("Furnishing Status:", ["Furnished", "Semi-furnished", "Unfurnished"])

with col2:
    mainroad = st.selectbox("Main Road Access:", ["Yes", "No"])
    guestroom = st.selectbox("Guest Room:", ["Yes", "No"])
    basement = st.selectbox("Basement:", ["Yes", "No"])
    hotwater = st.selectbox("Hot Water Heating:", ["Yes", "No"])
    airconditioning = st.selectbox("Air Conditioning:", ["Yes", "No"])
    prefarea = st.selectbox("Preferred Area:", ["Yes", "No"])

# ---- Encode categorical inputs ----
def encode_yesno(val):
    return 1 if val == "Yes" else 0

def encode_furnishing(val):
    return {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}.get(val, 0)

# ---- Create feature vector ----
features = np.array([[
    area,
    bedrooms,
    bathrooms,
    stories,
    encode_yesno(mainroad),
    encode_yesno(guestroom),
    encode_yesno(basement),
    encode_yesno(hotwater),
    encode_yesno(airconditioning),
    parking,
    encode_furnishing(furnished),
    encode_yesno(prefarea),
    house_age
]])

# ---- Dummy model for demonstration ----
# Let's simulate with a simple model trained on random data
np.random.seed(42)
X_dummy = np.random.randint(0, 10, size=(100, 13))
y_dummy = X_dummy @ np.array([10000, 15000, 20000, 12000, 5000, 8000, 7000, 9000, 10000, 4000, 6000, 5000, -1000]) + 50000

model = LinearRegression()
model.fit(X_dummy, y_dummy)

# ---- Predict ----
if st.button("🔍 Predict Price"):
    price = model.predict(features)[0]
    st.success(f"💰 **Estimated House Price: ₹{price:,.2f}**")

# ---- Footer ----
st.markdown("---")
st.markdown("*This model is for educational purposes. You can replace it with your trained model for real predictions.*")
