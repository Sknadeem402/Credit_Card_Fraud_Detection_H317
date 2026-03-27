import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Safe Plotly Import
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    try:
        with open("model/fraud_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model not found. Please upload the 'model' folder to GitHub.")
        st.stop()

fraud_model, scaler = load_model()

# Load Dataset (optional for EDA)
@st.cache_data
def load_data():
    try:
        return pd.read_csv("credit_card_fraud_dataset.csv")
    except:
        return None

df = load_data()

# Sidebar for Prediction
st.sidebar.header("💳 Transaction Details")
amount = st.sidebar.number_input("Transaction Amount (₹)", min_value=0.0, value=1250.0)
merchant_id = st.sidebar.number_input("Merchant ID", min_value=1, value=300)
city = st.sidebar.selectbox("Location", ["Hyderabad", "Bangalore", "Pune", "Mumbai", "Delhi"])
tx_type = st.sidebar.selectbox("Transaction Type", ["purchase", "refund", "online", "POS"])
hour = st.sidebar.slider("Hour", 0, 23, 15)
day_of_week = st.sidebar.slider("Day of Week", 0, 6, 2)
month = st.sidebar.slider("Month", 1, 12, 3)

predict_clicked = st.sidebar.button("🔍 Check for Fraud", type="primary")

# Main App
st.title("💳 Credit Card Fraud Detection")

tab1, tab2, tab3 = st.tabs(["Overview", "EDA", "Prediction"])

with tab1:
    st.write("Real-time Credit Card Fraud Detection using Random Forest Model.")

with tab2:
    st.subheader("Exploratory Data Analysis")
    if df is not None:
        st.success(f"✅ Dataset loaded successfully! ({len(df):,} transactions)")
        col1, col2 = st.columns(2)
        with col1:
            if PLOTLY_AVAILABLE:
                st.plotly_chart(px.histogram(df, x="Amount", title="Transaction Amount Distribution"), use_container_width=True)
        with col2:
            if PLOTLY_AVAILABLE:
                st.plotly_chart(px.pie(df, names="IsFraud", title="Fraud vs Legitimate"), use_container_width=True)
    else:
        st.warning("⚠️ Dataset not found.")
        st.info("Tip: Upload `credit_card_fraud_dataset.csv` to your GitHub repository to enable full EDA charts.")

with tab3:
    st.subheader("🔮 Make a Prediction")
    if predict_clicked:
        tx_enc = 1 if tx_type in ["purchase", "online", "POS"] else 0
        loc_enc = {"Hyderabad":0, "Bangalore":1, "Pune":2, "Mumbai":3, "Delhi":4}.get(city, 0)

        features = np.array([[amount, merchant_id, tx_enc, loc_enc, hour, day_of_week, month]])
        features_scaled = scaler.transform(features)

        pred = fraud_model.predict(features_scaled)[0]
        proba = fraud_model.predict_proba(features_scaled)[0][1]

        if pred == 1:
            st.error(f"🚨 FRAUD DETECTED! ({proba*100:.1f}% probability)")
        else:
            st.success(f"✅ Legitimate Transaction ({(1-proba)*100:.1f}% safe)")

        st.progress(proba, text=f"Fraud Risk: {proba*100:.1f}%")
    else:
        st.info("👈 Use the sidebar to enter details and click 'Check for Fraud'")

st.caption("Credit Card Fraud Detection | Random Forest Model")