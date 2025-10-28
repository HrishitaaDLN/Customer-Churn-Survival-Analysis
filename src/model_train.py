"""
model_train.py
--------------
Train multiple survival models:
- Cox Proportional Hazards (CoxPH)
- Weibull Accelerated Failure Time (AFT)
- Random Survival Forest (RSF)
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from .preprocess import preprocess_data
import joblib


# ============================================================
# 1. Cox Proportional Hazards (CoxPH)
# ============================================================

def train_coxph(df):
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if "customerID" in c], errors="ignore")

    df = df.select_dtypes(include=[np.number]).fillna(0)
    df = df.loc[:, df.var() > 0]

    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    df = df.drop(columns=to_drop, errors="ignore")

    cph = CoxPHFitter(penalizer=0.1)

    try:
        cph.fit(df, duration_col='tenure', event_col='Churn')
    except Exception as e:
        print("⚠️ CoxPH fitting failed due to collinearity — retrying reduced model:", e)
        reduced_cols = [c for c in df.columns if c not in ['tenure', 'Churn']][:15]
        cph.fit(df[['tenure', 'Churn'] + reduced_cols], duration_col='tenure', event_col='Churn')

    features = [col for col in df.columns if col not in ['Churn', 'tenure']]
    joblib.dump(features, "features.pkl")
    print(f"✅ Saved {len(features)} feature names for CoxPH.")
    return cph


# ============================================================
# 2. Weibull Accelerated Failure Time (AFT)
# ============================================================

def train_weibull_aft(df: pd.DataFrame):
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if "customerID" in c], errors="ignore")
    df = df.select_dtypes(include=[np.number]).fillna(0)
    df["tenure"] = df["tenure"].clip(lower=0.01)  # ✅ avoids non-positive durations

    aft = WeibullAFTFitter()
    aft.fit(df, duration_col="tenure", event_col="Churn")

    features = [col for col in df.columns if col not in ["Churn", "tenure"]]
    joblib.dump(features, "features.pkl")
    print(f"✅ Saved {len(features)} feature names for Weibull AFT.")
    return aft


# ============================================================
# 3. Random Survival Forest (RSF)
# ============================================================

def train_random_survival_forest(df: pd.DataFrame):
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if "customerID" in c], errors="ignore")

    X = df.drop(columns=["tenure", "Churn"], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    churn_bool = df["Churn"].astype(bool)
    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    y = np.array(list(zip(churn_bool, tenure)), dtype=[("event", "bool"), ("time", "f8")])

    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rsf.fit(X, y)

    joblib.dump(list(X.columns), "features.pkl")
    print(f"✅ Saved {len(X.columns)} RSF feature names for prediction alignment.")

    ci = concordance_index(df["tenure"], -rsf.predict(X), df["Churn"])
    print(f"Random Survival Forest Concordance Index: {ci:.3f}")
    return rsf
