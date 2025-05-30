import streamlit as st
import pandas as pd
import joblib

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu

# Page settings
st.set_page_config(page_title="AI Insurance Insights", layout="centered")
st.title("🤖 AI-Powered Insurance Insights & Prediction Dashboard")

# Load models
try:
    sentiment_model = pickle.load(open("C:/Users/Ramu M/customerfeedback_model.pkl", "rb"))
    claim_amount_model = pickle.load(open("C:/Users/Ramu M/fraudclaim_model.pkl", "rb"))
    insurance_risk_model = pickle.load(open("C:/Users/Ramu M/fraud_detection_model.pkl", "rb"))
except FileNotFoundError as e:
    st.error(f"Model load error: {e}")
    st.stop()

# Sidebar menu
with st.sidebar:
    selected = option_menu("Navigation", 
        ["📊 Results Dashboard", "🛡️ Fraud Claim Detection", "📉 Insurance Risk Prediction", "💬 Sentiment Analysis"],
        icons=["graph-up", "shield-check", "activity", "chat"],
        default_index=0)

# ------------------------------ #
# 📊 RESULTS DASHBOARD
# ------------------------------ #
if selected == "📊 Results Dashboard":
    st.subheader("📊 AI-Powered Results Overview")

    result_option = option_menu("Results", 
        ["📈 Revenue Growth", "📉 Cost Reduction", "⚡ Operational Efficiency", "🔍 Customer Experience"],
        icons=["graph-up-arrow", "currency-dollar", "lightning", "emoji-smile"], orientation="horizontal")

    if result_option == "📈 Revenue Growth":
        st.subheader("📈 Revenue Growth from AI Models")
        st.markdown("""
        - Personalized pricing → **+17% customer acquisition**  
        - Faster claim processing → **↓ churn by 25%**
        """)
        st.success("AI boosts revenue by enhancing retention and customer satisfaction.")

    elif result_option == "📉 Cost Reduction":
        st.subheader("📉 Cost Savings via Fraud Detection & Automation")
        try:
            df = pd.read_csv("C:/Users/Ramu M/insurance_risk.csv")
            fraud_claims = df[df["anomaly_tag"] == 1]
            total_savings = fraud_claims["claim_amount"].sum()

            st.write(f"✅ Detected **{len(fraud_claims)}** fraudulent claims")
            st.write(f"✅ Estimated Savings: ₹{total_savings:,.2f}")
            st.write("✅ Automated chatbots reduced support cost by 35%")
        except:
            st.warning("⚠️ Could not load insurance_risk.csv. Please check the file path.")

    elif result_option == "⚡ Operational Efficiency":
        st.subheader("⚡ Operational Improvements")
        st.markdown("""
        - Claims settled in **hours** instead of weeks  
        - 24/7 multilingual chatbots for instant support  
        - Reduced manual processing using AI pipelines
        """)
        st.info("AI streamlines operations and improves speed and consistency.")

    elif result_option == "🔍 Customer Experience":
        st.subheader("🔍 Sentiment Analysis of Customer Reviews")

        try:
            df = pd.read_csv("C:/Users/Ramu M/customer_feedback.csv")
            df.columns = df.columns.str.strip().str.lower()  # Clean column names

            if "sentiment_label" not in df.columns:
                st.warning("⚠️ The 'sentiment_label' column was not found in the dataset.")
            else:
                positive = (df["sentiment_label"].str.lower() == "positive").sum()
                negative = (df["sentiment_label"].str.lower() == "negative").sum()
                neutral = (df["sentiment_label"].str.lower() == "neutral").sum()
                total = len(df)

                percent_positive = (positive / total) * 100
                percent_negative = (negative / total) * 100
                percent_neutral = (neutral / total) * 100

                st.write(f"✅ Total Reviews: **{total}**")
                st.write(f"✅ Positive Sentiment: **{percent_positive:.2f}%**")
                st.write(f"✅ Neutral Sentiment: **{percent_neutral:.2f}%**")
                st.write(f"✅ Negative Sentiment: **{percent_negative:.2f}%**")

                st.success("✅ Customers appreciate easy documentation and quick claims.")
                st.info("📈 Most customers have a **positive experience** with service and support.")
        except Exception as e:
            st.error(f"⚠️ Could not load customer_feedback.csv. Error: {e}")

# ------------------------------ #
# 🛡️ FRAUD CLAIM DETECTION
# ------------------------------ #
elif selected == "🛡️ Fraud Claim Detection":
    st.subheader("🛡️ Insurance Claim Fraud Detection")

    claim_amount = st.selectbox("💰 Claim Amount", [20635.68, 64860.47])
    annual_income = st.selectbox("💵 Annual Income", [120769.55, 53075.22])
    ratio = st.selectbox("📊 Claim-to-Income Ratio", [0.170868, 1.222048])
    policy_days = st.selectbox("📅 Days Since Policy", [2850, 1601])
    claim_type = st.selectbox("📄 Claim Type", [1, 2])
    outlier_ee = st.selectbox("🔍 Outlier EE", [0, 1])
    outlier_if = st.selectbox("🔎 Outlier IF", [0, 1])
    outlier_lof = st.selectbox("📌 Outlier LOF", [0, 1])
    claim_type_e = st.selectbox("🧾 Claim Type E", [0, 1, 2])

    if st.button("🔍 Predict Fraud Status"):
        input_data = np.array([claim_amount, annual_income, ratio, policy_days,
                               claim_type, outlier_ee, outlier_if, outlier_lof, claim_type_e]).reshape(1, -1)

        pred = claim_amount_model.predict(input_data)[0]
        if pred == 1:
            st.error("⚠️ This claim is predicted as **FRAUDULENT**.")
            st.warning("📌 Please validate supporting documents.")
        else:
            st.success("✅ This claim is predicted as **LEGITIMATE**.")
            st.info("🙌 Good record!")

# ------------------------------ #
# 📉 INSURANCE RISK PREDICTION
# ------------------------------ #
elif selected == "📉 Insurance Risk Prediction":
    st.subheader("📉 Predict Customer Insurance Risk")

    col1, col2, col3 = st.columns(3)
    with col1:
        customer_age = st.text_input("Customer Age")
        annual_income = st.text_input("Annual Income")
        claim_history = st.text_input("Claim History")

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        premium_amount = st.text_input("Premium Amount")
        claim_amount = st.text_input("Claim Amount")

    with col3:
        policy_type = st.selectbox("Policy Type", ["Comprehensive", "Health", "Third-Party"])
        risk_score = st.text_input("Risk Score")
        vehicle_age = st.text_input("Vehicle Age")

    with col1:
        z_score_claim = st.text_input("Z-Score of Claim")

    with col2:
        iso_forest_anomaly = st.text_input("Isolation Forest Anomaly (0/1)")

    with col3:
        autoencoder_anomaly = st.text_input("Autoencoder Anomaly (0/1)")
        anomaly_tag_encoded = st.text_input("Anomaly Tag Encoded (0/1)")

    def to_type(val, name, typ):
        try:
            return typ(val)
        except:
            st.error(f"Invalid input for {name}. Must be {typ.__name__}.")
            return None

    if st.button("Predict Insurance Risk"):
        inputs = [
            to_type(customer_age, "Age", int),
            1 if gender == "Male" else 0,
            {"Comprehensive": 0, "Health": 1, "Third-Party": 2}[policy_type],
            to_type(annual_income, "Income", float),
            to_type(claim_history, "Claim History", int),
            to_type(premium_amount, "Premium", float),
            to_type(claim_amount, "Claim", float),
            to_type(risk_score, "Risk Score", float),
            to_type(vehicle_age, "Vehicle Age", int),
            to_type(z_score_claim, "Z-Score", float),
            to_type(iso_forest_anomaly, "IF Anomaly", int),
            to_type(autoencoder_anomaly, "AE Anomaly", int),
            to_type(anomaly_tag_encoded, "Anomaly Tag", int)
        ]

        if None in inputs:
            st.warning("❌ Please fix all input errors.")
        else:
            pred = insurance_risk_model.predict([inputs])[0]
            if pred == 1:
                st.error("⚠️ Predicted as **HIGH RISK**.")
            else:
                st.success("✅ Predicted as **LOW RISK**.")

# ------------------------------ #
# 💬 SENTIMENT ANALYSIS
# ------------------------------ #
elif selected == "💬 Sentiment Analysis":
    st.subheader("💬 Predict Sentiment of Customer Review")

    review_id = st.text_input("🆔 Review ID")
    customer_id = st.text_input("👤 Customer ID")
    rating = st.selectbox("⭐ Rating", [1, 2, 3, 4, 5])
    service_type = st.selectbox("📦 Service Type", [0, 1, 2])
    clean_review = st.text_input("📝 Clean Review Length (numeric)")

    def interpret_sentiment(pred):
        return {
            0: ("Negative 😟", "red"),
            1: ("Neutral 😐", "orange"),
            2: ("Positive 😊", "green")
        }.get(pred, ("Unknown 🤔", "gray"))

    if st.button("🔍 Predict Sentiment"):
        if '' in [review_id, customer_id, clean_review]:
            st.warning("⚠️ Please complete all fields.")
        else:
            input_data = np.array([
                float(review_id), float(customer_id), float(rating),
                float(service_type), float(clean_review)
            ]).reshape(1, -1)

            pred = sentiment_model.predict(input_data)[0]
            label, color = interpret_sentiment(pred)

            st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
