import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_fraud_model():
    try:
        with open("model/fraud_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run the training script first.")
        st.stop()

fraud_model, scaler = load_fraud_model()

# ==========================================================
# LOAD DATA FOR EDA
# ==========================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("credit_card_fraud_dataset.csv")
        return df
    except:
        return None

df = load_data()

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.markdown("### 💳 Transaction Details")

amount = st.sidebar.number_input("Transaction Amount (₹)", min_value=0.0, value=850.0, step=10.0)
merchant_id = st.sidebar.number_input("Merchant ID", min_value=1, value=245, step=1)

city = st.sidebar.selectbox("📍 Location", ["Hyderabad", "Bangalore", "Pune", "Mumbai", "Delhi"])
tx_type = st.sidebar.selectbox("🛍️ Transaction Type", ["purchase", "refund", "online", "POS"])

hour = st.sidebar.slider("⏰ Hour of Transaction", 0, 23, 14)
day_of_week = st.sidebar.slider("📅 Day of Week (0=Mon)", 0, 6, 2)
month = st.sidebar.slider("📆 Month", 1, 12, 3)

predict_clicked = st.sidebar.button("🔍 Check for Fraud", type="primary", use_container_width=True)

# ==========================================================
# MAIN TITLE
# ==========================================================
st.title("💳 Credit Card Fraud Detection System")
st.markdown("Real-time Fraud Detection using Machine Learning")

# Navigation
if "page" not in st.session_state:
    st.session_state.page = "overview"

col1, col2, col3, col4 = st.columns(4)
if col1.button("🏠 Overview", use_container_width=True): st.session_state.page = "overview"
if col2.button("📊 Dataset", use_container_width=True): st.session_state.page = "dataset"
if col3.button("🔍 EDA", use_container_width=True): st.session_state.page = "eda"
if col4.button("🔮 Prediction", use_container_width=True): st.session_state.page = "prediction"

st.markdown("---")

# ==========================================================
# PAGES
# ==========================================================
if st.session_state.page == "overview":
    st.markdown("### Project Overview")
    st.write("""
    This application uses a **Random Forest Classifier** to detect fraudulent credit card transactions in real-time.
    
    It includes:
    - Real-time Fraud Prediction
    - Dataset Overview
    - Exploratory Data Analysis (EDA)
    """)

elif st.session_state.page == "dataset":
    st.markdown("### 📊 Dataset Information")
    if df is not None:
        st.write(f"**Total Transactions:** {len(df):,}")
        st.write(f"**Fraud Rate:** {df['IsFraud'].mean()*100:.2f}%")
        st.dataframe(df.head(), use_container_width=True)
    else:
        st.warning("Dataset not found.")

elif st.session_state.page == "eda":
    st.markdown("### 🔍 Exploratory Data Analysis")
    if df is not None:
        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.histogram(df, x="Amount", nbins=50, title="Transaction Amount Distribution")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.pie(df, names="IsFraud", title="Fraud vs Legitimate Transactions")
            st.plotly_chart(fig2, use_container_width=True)

        # Fraud by Location
        fig3 = px.bar(df.groupby("Location")["IsFraud"].mean().reset_index(), 
                      x="Location", y="IsFraud", 
                      title="Fraud Rate by Location")
        st.plotly_chart(fig3, use_container_width=True)

        # Amount vs Fraud
        fig4 = px.box(df, x="IsFraud", y="Amount", title="Transaction Amount by Fraud Status")
        st.plotly_chart(fig4, use_container_width=True)

    else:
        st.warning("Dataset not available for EDA.")

# ==========================================================
# PREDICTION PAGE
# ==========================================================
elif st.session_state.page == "prediction":
    st.markdown("### 🔮 Fraud Detection Result")

    if predict_clicked:
        # Encode inputs
        tx_type_enc = 1 if tx_type in ["purchase", "online", "POS"] else 0
        location_enc = {"Hyderabad":0, "Bangalore":1, "Pune":2, "Mumbai":3, "Delhi":4}.get(city, 0)

        features = np.array([[amount, merchant_id, tx_type_enc, location_enc, 
                              hour, day_of_week, month]])

        features_scaled = scaler.transform(features)
        prediction = fraud_model.predict(features_scaled)[0]
        proba = fraud_model.predict_proba(features_scaled)[0][1]

        if prediction == 1:
            st.error(f"🚨 **FRAUDULENT TRANSACTION DETECTED!**")
            st.metric("Fraud Probability", f"{proba*100:.1f}%", delta="High Risk")
        else:
            st.success(f"✅ **LEGITIMATE TRANSACTION**")
            st.metric("Legitimate Probability", f"{(1-proba)*100:.1f}%", delta="Safe")

        st.progress(proba, text=f"Fraud Risk: {proba*100:.1f}%")

        # Gauge Chart
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": "Fraud Probability (%)"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "red" if proba > 0.5 else "green"}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

    else:
        st.info("👈 Enter details in sidebar and click **Check for Fraud**")

# Footer
st.markdown("---")
st.caption("Credit Card Fraud Detection System | Random Forest + Streamlit")