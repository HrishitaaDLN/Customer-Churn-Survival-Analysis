# Customer Churn Survival Analysis

This project reframes customer churn prediction as a **time-to-event (survival)** problem — focusing on **when** a customer will churn, not just **if** they will.  

It combines both **statistical** and **machine learning** approaches — including **Cox Proportional Hazards**, **Weibull Accelerated Failure Time**, and **Random Survival Forests** — to estimate churn risk, customer lifetime, and survival probabilities for telecom customers.

You can view the app here: https://customer-churn-survival-analysis-ldhar.streamlit.app/
---

##  1. Project Overview

Traditional churn prediction treats churn as a binary classification problem.  
This project models churn as a **survival process**, estimating the **expected time until churn**.

This allows:
- Early intervention for at-risk customers  
- More accurate retention & revenue forecasting  
- Data-driven customer lifetime value (CLV) estimation  

The analysis uses the **IBM Telco Customer Churn dataset** as a case study.

---

##  2. Objectives

- Estimate **customer survival probability** over time  
- Identify **key drivers of churn risk**  
- Quantify the **hazard (instantaneous risk)** of churn  
- Compare **interpretable (CoxPH)** and **ML-based (RSF)** survival models  
- Translate findings into **business recommendations**  

---

##  3. Methodology

###  Data Exploration
- Visualized churn patterns by demographics, services, and tenure  
- Detected early-tenure churn risk  
- Checked feature multicollinearity using **Variance Inflation Factor (VIF)**

###  Feature Engineering
- One-hot encoded categorical variables  
- Removed high-VIF features (>5)  
- Defined:
  - `tenure` → customer lifetime (duration)
  - `Churn` → churn event indicator (1 = churned, 0 = retained)

---

##  4. Models Implemented

| Model | Type | Purpose |
|--------|------|----------|
| Kaplan–Meier Estimator | Non-parametric | Baseline survival probability |
| Cox Proportional Hazards | Semi-parametric | Measures relative churn risk |
| Weibull AFT Model | Parametric | Models time to churn |
| Random Survival Forest | Ensemble | Captures nonlinear feature effects |

---

##  5. Key Results

| Metric | Value | Interpretation |
|--------|--------|----------------|
| Concordance Index (Weibull AFT) | **0.87** | Excellent predictive accuracy |
| AIC (Weibull) | **17,877** | Strong model fit |
| Mean Survival Time | **37.4 months** | Avg. customer lifetime |
| Median Survival Time | **24.2 months** | 50% churn by ~2 years |

---

##  6. Insights & Interpretations

###  Early Churn Risk
- Highest churn risk within **first 6 months**  
- Retaining customers past **12 months** dramatically improves lifetime

###  Service & Payment Behavior
- **Month-to-month contracts** and **electronic check payments** drive churn  
- **Long-term contracts** & **auto-pay** reduce risk

###  Add-on Services
- **OnlineSecurity**, **TechSupport**, and **OnlineBackup** improve retention  
- Bundled services consistently extend customer lifetime

###  Feature Importance (RSF)
Top churn drivers:
1. Contract type (Month-to-month)
2. Payment method (Electronic check)
3. Monthly charges
4. Internet service type

Demographics (e.g., gender, senior status) showed minimal effect.

---

##  7. Model Comparison

| Model | Concordance Index | Notes |
|--------|------------------|-------|
| Kaplan–Meier | — | Baseline non-parametric |
| CoxPH | 0.83 | Interpretable hazard ratios |
| Weibull AFT | **0.87** | Best overall performer |
| Random Survival Forest | 0.81 | Captures non-linear effects |
| CoxNet (Lasso) | 0.84 | Adds regularization |

---

##  8. Business Recommendations

| Finding | Recommendation |
|----------|----------------|
| Early churn (0–12 months) | Improve onboarding & early engagement |
| Long-term contracts increase retention | Incentivize 1–2 year commitments |
| Electronic check users churn faster | Encourage credit/auto-pay |
| Bundled services improve retention | Promote TechSupport & OnlineSecurity |
| Seniors churn faster | Offer loyalty discounts or personalized support |

---

##  9. Business Impact

Assuming **10,000 customers** at **\$70/month**:
- Extending survival by **3 months** = **\$2.1M retained revenue annually**
- Enables **timely, targeted retention campaigns**

---

##  10. Technical Details

**Language:** Python 3.10+  
**Environment:** Jupyter Notebook / Streamlit  

**Core Libraries:**  
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `lifelines`, `scikit-survival`, `joblib`, `fpdf`

---

##  11. Running the Project

### Clone the repository
```bash
git clone https://github.com/HrishitaaDLN/Customer-Churn-Survival-Analysis.git
cd Customer-Churn-Survival-Analysis
Install dependencies
pip install -r requirements.txt

Run the Streamlit app
streamlit run streamlit_app.py


Upload the dataset (or use the default Telco dataset) → select a model → train → analyze churn curves, feature importances, and predictions.

## 12. Author

Lakshmi Naga Hrishitaa Dharmavarapu
Machine Learning & Survival Modeling
 Email: ldhar@uic.edu

 LinkedIn: www.linkedin.com/in/hrishitaa-dharmavarapu-ln-3420a8205

 GitHub: https://github.com/HrishitaaDLN

## 13. Acknowledgments

Dataset: IBM Telco Customer Churn

Libraries: Lifelines, Scikit-Survival, PyCox

Inspiration: Survival analysis applied to customer retention

## 14. Summary

This project demonstrates how survival analysis transforms churn prediction —
revealing not only who is likely to churn, but when it will happen.

By combining interpretable statistics with robust machine learning,
it provides both predictive precision and actionable business insights.
