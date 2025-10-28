"""
utils.py
--------
Utility functions for survival model evaluation and diagnostics.

Includes:
- Variance Inflation Factor (VIF)
- Feature importance plotting
- Kaplan–Meier curve visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lifelines import KaplanMeierFitter
import numpy as np


# ==============================================================
# 1. Variance Inflation Factor (VIF)
# ==============================================================

def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for each numeric feature.
    High VIF (>5) indicates multicollinearity risk.
    """
    X = df.select_dtypes(include=['number']).dropna()
    if X.empty:
        st.warning("⚠️ No numeric features found for VIF calculation.")
        return pd.DataFrame(columns=["Feature", "VIF"])

    vif_data = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_data.sort_values("VIF")


# ==============================================================
# 2. Kaplan-Meier Survival Curve
# ==============================================================

def plot_km_curve(df: pd.DataFrame, time_col: str, event_col: str, label: str = "All Customers"):
    """
    Plot Kaplan-Meier survival curve with validation and clean rendering.
    """
    try:
        # Ensure numeric + valid data
        df = df.copy()
        df = df[[time_col, event_col]].dropna()
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        df[event_col] = df[event_col].astype(int)

        if df.empty or df[time_col].isnull().all():
            st.warning("⚠️ No valid data available to plot survival curve.")
            return None

        kmf = KaplanMeierFitter()
        kmf.fit(df[time_col], event_observed=df[event_col], label=label)

        fig, ax = plt.subplots(figsize=(7, 5))
        kmf.plot(ax=ax, color="navy", ci_show=True)
        ax.set_title(f"Kaplan–Meier Survival Curve: {label}")
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Retention Probability")
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Error plotting KM curve: {e}")
        return None

# ==============================================================
# 3. Feature Importance Plot (RSF, Cox, AFT)
# ==============================================================

def plot_feature_importance(model):
    try:
        if hasattr(model, "feature_importances_") and getattr(model, "feature_importances_", None) is not None:
            cols = getattr(model, "training_columns_", None)
            importances = pd.Series(model.feature_importances_, index=cols)
            importances = importances.sort_values(ascending=True).tail(15)
            fig, ax = plt.subplots(figsize=(6, 5))
            importances.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title("Top Feature Importances")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            return fig
        else:
            st.warning("⚠️ Model does not expose feature importances.")
            return None
    except Exception as e:
        st.error(f"Error plotting feature importance: {e}")
        return None
