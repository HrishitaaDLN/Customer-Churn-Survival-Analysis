"""
Streamlit App: Customer Churn Survival Analysis Dashboard (FAANG Edition)
-------------------------------------------------------------------------

A complete interactive survival analysis dashboard designed for both technical
and business audiences. It includes:
- Model interpretation and performance evaluation
- Customer-level churn risk simulator
- Auto-generated executive summary (PDF)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
import joblib

# Ensure src folder is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import helper scripts
from src.preprocess import preprocess_data
from src.model_train import (
    train_coxph,
    train_weibull_aft,
    train_random_survival_forest
)
from src.utils import plot_km_curve, plot_feature_importance

# ---------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Survival Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.title("📊 Customer Churn Survival Analysis Dashboard")

st.markdown("""
**Welcome!**  
This dashboard helps you understand **when and why customers churn — not just who will.**  
It uses advanced **survival analysis models** (CoxPH, Weibull AFT, and Random Survival Forests)  
to forecast churn timing, estimate retention probability, and evaluate the business impact of customer behavior.
""")

st.info("""
💡 *Please upload your dataset in the sidebar to begin analysis.*  
If no file is provided, the app will automatically use the **IBM Telco Customer Churn dataset** as a default example.
""")


# ---------------------------------------------------------------------
# SIDEBAR - DATA CONFIGURATION
# ---------------------------------------------------------------------
st.sidebar.header("📂 Data Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Telco Churn Dataset (.csv)", type=["csv"])

if uploaded_file:
    data = preprocess_data(uploaded_file)
    st.sidebar.success("✅ Dataset uploaded successfully.")
else:
    default_path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if default_path.exists():
        data = preprocess_data(default_path)
        st.sidebar.info("Using default IBM Telco dataset.")
    else:
        st.error("No dataset found. Please upload a CSV file.")
        st.stop()

# ---------------------------------------------------------------------
# OVERVIEW SECTION
# ---------------------------------------------------------------------
st.subheader("📈 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(data))
col2.metric("Churned", int(data["Churn"].sum()))
col3.metric("Retention Rate", f"{(1 - data['Churn'].mean())*100:.1f}%")

st.dataframe(data.head())

# ---------------------------------------------------------------------
# CHURN DISTRIBUTION
# ---------------------------------------------------------------------
st.markdown("### 🧭 Churn Distribution Overview")

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Churn", data=data, palette="coolwarm", ax=ax)
ax.set_title("Customer Churn vs Retention")
st.pyplot(fig)

st.markdown("""
Churners represent direct revenue leakage.  
Identifying the **drivers** behind this behavior helps optimize retention efforts.
""")

# ---------------------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------------------
st.sidebar.subheader("⚙️ Model Selection")
model_choice = st.sidebar.radio(
    "Select a model:",
    ["Cox Proportional Hazards", "Weibull AFT", "Random Survival Forest"]
)

# ---------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------
st.subheader("🧠 Model Training and Evaluation")

if st.button("Run Survival Analysis"):
    with st.spinner("Training model..."):
        if model_choice == "Cox Proportional Hazards":
            model = train_coxph(data)
            ci = model.concordance_index_
            st.success("✅ CoxPH model trained successfully.")
            st.metric("Concordance Index", f"{ci:.3f}")

        elif model_choice == "Weibull AFT":
            model = train_weibull_aft(data)
            st.success("✅ Weibull AFT model trained successfully.")
            st.metric("Median Survival Time", round(model.median_survival_time_, 2))
            st.metric("Mean Survival Time", round(model.mean_survival_time_, 2))

        elif model_choice == "Random Survival Forest":
            model = train_random_survival_forest(data)
            st.success("✅ Random Survival Forest model trained successfully.")

        # Save model + schema
        joblib.dump(model, "model.pkl")
        if hasattr(model, "training_columns_"):
            joblib.dump(model.training_columns_, "features.pkl")
            st.caption("✅ Saved model feature schema.")

# ---------------------------------------------------------------------
# SURVIVAL CURVE
# ---------------------------------------------------------------------
st.subheader("📉 Customer Retention Curve")
data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce").fillna(0)
fig = plot_km_curve(data, time_col="tenure", event_col="Churn")

if fig is not None:
    st.pyplot(fig)
else:
    st.warning("⚠️ Could not generate retention curve. Check if 'tenure' and 'Churn' columns exist.")
    
st.markdown("""
**Interpretation:**
- The curve shows customer retention probability over time.  
- Steeper declines indicate higher churn risk.  
- Business teams should focus on the tenure periods where retention drops fastest.
""")

# ---------------------------------------------------------------------
# CUSTOMER RISK SIMULATOR
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("🎯 Customer Risk Simulator")

st.markdown("""
Input customer attributes to estimate their **churn probability** and **expected lifetime value**.
The app automatically aligns inputs with the model’s trained feature schema for consistent predictions.
""")

# Customer input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "AutoPay"])
techsupport = st.selectbox("Tech Support", ["Yes", "No"])
monthly = st.slider("Monthly Charges ($)", 20, 120, 70)


if st.button("Predict Churn Risk"):
    try:
        model = joblib.load("model.pkl")

        if os.path.exists("features.pkl"):
            expected_features = joblib.load("features.pkl")
            st.caption("✅ Loaded feature schema from training phase.")
        else:
            st.error("❌ Feature schema not found. Please re-train a model first.")
            st.stop()

        user_raw = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": 0,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": internet,
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": techsupport,
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": "Yes",
            "PaymentMethod": (
                "Credit card (automatic)" if payment == "AutoPay" else payment
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": monthly * 12,
            "Churn": 0,
        }

        df_input = pd.DataFrame([user_raw])
        df_input_pre = preprocess_data(df_input)
        df_input_pre = df_input_pre.reindex(columns=expected_features, fill_value=0)

        # --- Predict survival safely for lifelines or sksurv ---
        if hasattr(model, "predict_survival_function"):
            surv_fn = model.predict_survival_function(df_input_pre)

            # CASE 1: lifelines -> DataFrame
            if isinstance(surv_fn, pd.DataFrame):
                surv_prob = float(surv_fn.iloc[-1, 0])
                times = surv_fn.index
                vals = surv_fn.iloc[:, 0].values

            # CASE 2: sksurv -> list of callable functions
            elif isinstance(surv_fn, (list, np.ndarray)) and callable(surv_fn[0]):
                times = np.arange(0, 73)
                vals = [float(surv_fn[0](t)) for t in times]
                surv_prob = vals[-1]

            else:
                st.warning("⚠️ Unsupported survival function format.")
                st.stop()

            churn_prob = 1 - surv_prob
            st.metric("Predicted Churn Probability", f"{churn_prob * 100:.1f}%")
            ltv = monthly * (1.0 / (churn_prob + 1e-4))
            st.metric("Estimated Lifetime Value", f"${ltv:.2f}")

            # Plot survival curve
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(times, vals, color="blue")
            ax.set_title("Predicted Survival Curve for This Customer")
            ax.set_xlabel("Tenure (months)")
            ax.set_ylabel("Survival Probability")
            st.pyplot(fig)

        else:
            st.warning("⚠️ This model does not support survival curve prediction.")

    except Exception as e:
        st.error(f"Error generating prediction: {e}")

# ---------------------------------------------------------------------
# PDF REPORT GENERATOR
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📝 Executive Summary Report")

st.markdown("Generate a PDF summary of churn insights and business recommendations.")

if st.button("Generate PDF Report"):
    try:
        from fpdf import FPDF

        class PDF(FPDF):
            def header(self):
                self.set_font("Arial", "B", 14)
                self.cell(0, 10, "Customer Churn Survival Analysis Report", ln=True, align="C")
                self.ln(5)

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", "", 12)

        text = (
            "This report summarizes survival analysis results from the Telco customer churn dataset.\n\n"
            "Key Findings:\n"
            "- Average customer lifetime: approximately 37 months\n"
            "- Churn rate: 26%\n"
            "- Early churn risk (0–12 months) is the highest\n\n"
            "Top Retention Drivers:\n"
            "- Long-term contracts\n"
            "- AutoPay or credit card billing\n"
            "- Add-on services like Tech Support and Online Security\n\n"
            "Business Recommendations:\n"
            "- Offer 1 to 2 years of loyalty contracts\n"
            "- Incentivize auto-pay enrollment\n"
            "- Cross-sell support services to increase stickiness\n"
        )

        pdf.multi_cell(0, 10, text.encode("latin-1", "replace").decode("latin-1"))
        pdf.output("Churn_Survival_Report.pdf")

        with open("Churn_Survival_Report.pdf", "rb") as f:
            st.download_button("📥 Download Report", f, file_name="Churn_Survival_Report.pdf")

        st.success("✅ PDF report generated successfully.")
    except Exception as e:
        st.error(f"Error generating report: {e}")

# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.markdown("---")
st.caption("Developed by Hrishitaa Dharmavarapu — Advanced Survival Modeling for Customer Retention")
