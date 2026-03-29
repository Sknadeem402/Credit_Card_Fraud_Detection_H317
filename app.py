import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

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

# ── FIX 1: Use absolute path relative to this script's location ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "model", "scaler.pkl")
CSV_PATH     = os.path.join(BASE_DIR, "credit_card_fraud_dataset.csv")

# Load Model
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Model not found: {e}")
        st.info(f"Expected model at: {MODEL_PATH}")
        st.stop()

fraud_model, scaler = load_model()

# ── FIX 2: Show actual error + allow in-app CSV upload as fallback ──
@st.cache_data
def load_data(csv_path=CSV_PATH):
    try:
        df = pd.read_csv(csv_path)
        return df, None          # (dataframe, error_message)
    except FileNotFoundError:
        return None, f"CSV not found at: {csv_path}"
    except Exception as e:
        return None, str(e)

df, load_error = load_data()

# Sidebar for Prediction
st.sidebar.header("💳 Transaction Details")
amount       = st.sidebar.number_input("Transaction Amount (₹)", min_value=0.0, value=1250.0)
merchant_id  = st.sidebar.number_input("Merchant ID", min_value=1, value=300)
city         = st.sidebar.selectbox("Location", ["Hyderabad", "Bangalore", "Pune", "Mumbai", "Delhi"])
tx_type      = st.sidebar.selectbox("Transaction Type", ["purchase", "refund", "online", "POS"])
hour         = st.sidebar.slider("Hour", 0, 23, 15)
day_of_week  = st.sidebar.slider("Day of Week", 0, 6, 2)
month        = st.sidebar.slider("Month", 1, 12, 3)

predict_clicked = st.sidebar.button("🔍 Check for Fraud", type="primary")

# Main App
st.title("💳 Credit Card Fraud Detection")

tab1, tab2, tab3 = st.tabs(["Overview", "EDA", "Prediction"])

with tab1:
    st.write("Real-time Credit Card Fraud Detection using Random Forest Model.")

with tab2:
    st.subheader("Exploratory Data Analysis")

    if df is not None:
        st.success(f"✅ Dataset loaded! ({len(df):,} transactions)")
        col1, col2 = st.columns(2)
        with col1:
            if PLOTLY_AVAILABLE:
                st.plotly_chart(
                    px.histogram(df, x="Amount", title="Transaction Amount Distribution"),
                    use_container_width=True
                )
        with col2:
            if PLOTLY_AVAILABLE:
                st.plotly_chart(
                    px.pie(df, names="IsFraud", title="Fraud vs Legitimate"),
                    use_container_width=True
                )
        st.dataframe(df.head(10), use_container_width=True)

    else:
        # ── Show the actual error so user knows what went wrong ──
        st.warning(f"⚠️ Could not load dataset automatically.")
        st.error(f"Reason: {load_error}")
        st.markdown(
            f"**Expected path:** `{CSV_PATH}`\n\n"
            "Make sure `credit_card_fraud_dataset.csv` is in the **same folder as `app.py`**."
        )

        # ── Fallback: let user upload the CSV directly in the browser ──
        st.divider()
        st.subheader("📂 Upload Dataset Manually")
        uploaded_file = st.file_uploader("Upload credit_card_fraud_dataset.csv", type=["csv"])
        if uploaded_file:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                st.success(f"✅ Uploaded! ({len(df_uploaded):,} rows)")
                col1, col2 = st.columns(2)
                with col1:
                    if PLOTLY_AVAILABLE:
                        st.plotly_chart(
                            px.histogram(df_uploaded, x="Amount", title="Transaction Amount Distribution"),
                            use_container_width=True
                        )
                with col2:
                    if PLOTLY_AVAILABLE:
                        st.plotly_chart(
                            px.pie(df_uploaded, names="IsFraud", title="Fraud vs Legitimate"),
                            use_container_width=True
                        )
                st.dataframe(df_uploaded.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")

with tab3:
    st.subheader("🔮 Make a Prediction")
    if predict_clicked:
        tx_enc  = 1 if tx_type in ["purchase", "online", "POS"] else 0
        loc_enc = {"Hyderabad": 0, "Bangalore": 1, "Pune": 2, "Mumbai": 3, "Delhi": 4}.get(city, 0)

        features        = np.array([[amount, merchant_id, tx_enc, loc_enc, hour, day_of_week, month]])
        features_scaled = scaler.transform(features)

        pred  = fraud_model.predict(features_scaled)[0]
        proba = fraud_model.predict_proba(features_scaled)[0][1]

        if pred == 1:
            st.error(f"🚨 FRAUD DETECTED! ({proba*100:.1f}% probability)")
        else:
            st.success(f"✅ Legitimate Transaction ({(1-proba)*100:.1f}% safe)")

        st.progress(proba, text=f"Fraud Risk: {proba*100:.1f}%")
    else:
        st.info("👈 Use the sidebar to enter details and click 'Check for Fraud'")

st.caption("Credit Card Fraud Detection | Random Forest Model")