import streamlit as st
import numpy as np
import pickle

# Load model
with open("models/house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Page config ---
st.set_page_config(page_title=" House Price Predictor", page_icon="", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
        .main {
            background-color: black;
        }
        .stApp {
            max-width: 800px;
            margin: auto;
            padding: 2rem;
            background: white;
            border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        h1 {
            color: #155DFC;
            text-align: center;
        }
        .css-1v3fvcr {
            background-color: #e0e7ff;
        }
        .css-2trqyj {
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title & Description ---
st.title(" Boston House Price Predictor")
st.markdown("""
Enter the property details below to estimate the **price of a house** in Boston using a machine learning model.  
""")

st.markdown("---")

# --- Input Section ---
st.header("📝 Property Features")

col1, col2 = st.columns(2)

with col1:
    CRIM = st.number_input("🔸 CRIM (Crime rate)", value=0.1)
    ZN = st.number_input("🔸 ZN (Zoned land %)", value=0.0)
    INDUS = st.number_input("🔸 INDUS (Industrial area %)", value=5.0)
    CHAS = st.selectbox("🔸 CHAS (Near Charles River?)", [0, 1])
    NOX = st.number_input("🔸 NOX (Nitric Oxides)", value=0.5)
    RM = st.number_input("🔸 RM (Avg rooms per dwelling)", value=6.0)
    AGE = st.number_input("🔸 AGE (% units built before 1940)", value=30.0)

with col2:
    DIS = st.number_input("🔹 DIS (Distance to employment centers)", value=5.0)
    RAD = st.number_input("🔹 RAD (Highway access index)", value=1.0)
    TAX = st.number_input("🔹 TAX (Property-tax rate)", value=300.0)
    PTRATIO = st.number_input("🔹 PTRATIO (Pupil-teacher ratio)", value=15.0)
    B = st.number_input("🔹 B (1000(Bk - 0.63)^2)", value=390.0)
    LSTAT = st.number_input("🔹 LSTAT (% lower status)", value=12.0)

# --- Prediction ---
st.markdown("----")

if st.button("💰 Predict House Price"):
    input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM,
                            AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    
    price_usd = model.predict(input_data)[0] * 1000  # USD
    price_inr = price_usd * 83  # Rough conversion to INR

    st.success(f" Estimated House Price: ${price_usd:,.2f} USD")
    st.info(f"🇮🇳 Equivalent in INR: ₹{price_inr:,.0f}")


# --- Footer ---
st.markdown("---")
st.markdown(
    "<center>Made with ❤️ using Streamlit & Scikit-Learn </center>",
    unsafe_allow_html=True
)
