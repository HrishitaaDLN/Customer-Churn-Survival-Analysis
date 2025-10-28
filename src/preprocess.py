"""
preprocess.py
--------------
Data cleaning and feature engineering for Survival Analysis.

This module:
- Loads raw CSV data
- Encodes categorical variables
- Converts churn and tenure to event/time format
- Prepares the dataset for survival modeling
"""

import pandas as pd
import numpy as np


def preprocess_data(file_or_df):
    if isinstance(file_or_df, str) or hasattr(file_or_df, "read"):
        df = pd.read_csv(file_or_df)
    else:
        df = file_or_df.copy()

    # Drop duplicates, clean column names
    df.columns = df.columns.str.strip()

    # Handle missing TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # Encode churn if exists
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

    # Get dummies for categorical features
    df = pd.get_dummies(df, drop_first=True)

    return df

if __name__ == "__main__":
    data = preprocess_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(data.head())
