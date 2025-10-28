# Customer Churn Survival Analysis

This project reframes customer churn prediction as a **time-to-event (survival)** problem — focusing on **when** a customer will churn, not just **if** they will.  
It combines statistical and machine learning methods — including **Cox Proportional Hazards**, **Weibull Accelerated Failure Time**, and **Random Survival Forests** — to estimate churn risk and customer survival probabilities for telecom customers.

---

## 1. Project Overview

Traditional churn prediction treats churn as a binary classification problem.  
Here, churn is modeled as a **survival analysis** problem that estimates the **expected time until churn**, enabling:
- Early intervention for at-risk customers  
- Better retention and revenue forecasting  
- Data-driven customer lifetime value (CLV) estimation  

The project leverages the **Telco Customer Churn dataset** to explore and model churn timing.

---

## 2. Objectives

- Estimate customer survival probability over time  
- Identify the key features driving churn risk and timing  
- Quantify the hazard (instantaneous risk) of churn  
- Compare interpretable and ML-based survival models  
- Provide actionable, business-ready recommendations  

---

## 3. Methodology

### Data Exploration
- Visualized churn by demographics, services, and tenure  
- Analyzed early-tenure churn risk  
- Assessed collinearity using Variance Inflation Factor (VIF)

### Feature Engineering
- One-hot encoded categorical variables  
- Removed high-VIF features to avoid redundancy  
- Defined:
  - `time`: customer tenure (duration)
  - `event`: churn indicator (1 = churned, 0 = retained)

### Models Implemented

| Model | Type | Purpose |
|--------|------|----------|
| Kaplan–Meier Estimator | Non-parametric | Baseline survival estimation |
| Cox Proportional Hazards | Semi-parametric | Measures relative churn risk |
| Weibull AFT Model | Parametric | Models accelerated/decelerated time to churn |
| CoxNet (Lasso) | Penalized | Adds feature selection and regularization |
| Random Survival Forest | Ensemble | Captures non-linear effects and interactions |
| (Optional) DeepSurv | Neural | Deep model for complex hazard estimation |

---

## 4. Key Results

| Metric | Value | Interpretation |
|--------|--------|----------------|
| Concordance Index (Weibull AFT) | 0.87 | Strong predictive accuracy |
| AIC (Weibull) | 17,877 | Good model fit |
| Mean Survival Time | 37.4 months | Average customer lifetime |
| Median Survival Time | 24.2 months | 50% churn by ~2 years |

---

## 5. Main Insights

### Early Churn Risk
- The highest churn rate occurs within the first **6 months**.  
- Retaining customers beyond **12 months** significantly improves survival.

### Service & Payment Behavior
- **Month-to-month contracts** and **electronic check payments** increase churn risk.  
- Customers with **one-year/two-year contracts** or **auto-pay** methods stay longer.

### Value-Added Services
- Services like **OnlineSecurity**, **TechSupport**, and **OnlineBackup** reduce churn.  
- Bundled services improve retention.

### Feature Importance (RSF)
Top drivers of churn:
1. Contract type (Month-to-month)
2. Payment method (Electronic check)
3. Monthly charges
4. Internet service type

Demographic features such as gender or senior status showed minimal influence.

---

## 6. Model Comparison

| Model | Concordance Index | Notes |
|--------|------------------|-------|
| Kaplan–Meier | — | Baseline model |
| CoxPH | 0.83 | Interpretable hazard ratios |
| Weibull AFT | 0.87 | Best overall performance |
| Random Survival Forest | 0.81 | Captures non-linear effects |
| CoxNet (Lasso) | 0.84 | Stable, regularized model |

---

## 7. Business Recommendations

| Finding | Recommendation |
|----------|----------------|
| Early churn (0–12 months) | Strengthen onboarding and engagement |
| Long contracts improve retention | Incentivize 1–2 year commitments |
| Electronic check users churn faster | Encourage auto-pay or credit card billing |
| Bundled services reduce churn | Promote TechSupport and OnlineSecurity packages |
| Senior customers churn faster | Offer targeted retention incentives |

---

## 8. Business Impact

Assuming 10,000 customers paying \$70/month:
- Extending average survival by **3 months** yields **\$2.1M in retained annual revenue**.  
- Predictive survival models enable **timely, targeted retention campaigns**.

---

## 9. Technical Details

**Language:** Python 3.10+  
**Environment:** Jupyter Notebook / Streamlit / Google Colab  
**Core Libraries:**  
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `lifelines`, `scikit-survival`, `pycox`, `torch`

---

## 10. Running the Project

Clone the repository:
```bash
git clone https://github.com/<your-username>/customer-churn-survival-analysis.git
cd customer-churn-survival-analysis
Install dependencies:

pip install -r requirements.txt


Launch the notebook:

jupyter notebook notebooks/SurvivalAnalysis_Final.ipynb


(Optional) Run the Streamlit app:

streamlit run app/streamlit_app.py

11. Future Work

Add DeepSurv for non-linear hazard estimation

Develop real-time churn risk dashboard

Deploy survival models as REST APIs

Integrate survival outputs with Customer Lifetime Value (CLV) prediction

12. Author

Your Name
Machine Learning & Survival Modeling
Email: your.email@example.com

LinkedIn: linkedin.com/in/yourname

GitHub: github.com/yourusername

13. Acknowledgments

Dataset: IBM Telco Customer Churn

Libraries: lifelines, scikit-survival, pycox

14. Summary

This project demonstrates how survival analysis can uncover when churn is most likely to occur, not just who will churn.
By integrating interpretable statistical models with robust ML methods, it provides both predictive precision and strategic business value.