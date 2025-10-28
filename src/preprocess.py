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
import io
from pathlib import Path


def preprocess_data(file_or_df):
    """
    Flexible preprocessing for Telco Churn data.
    Accepts: file path (str/Path), uploaded file, or existing DataFrame.
    Returns: cleaned DataFrame ready for survival modeling.
    """

    # ----------------------------------------------------------
    #  Load data depending on input type
    # ----------------------------------------------------------
    if isinstance(file_or_df, pd.DataFrame):
        df = file_or_df.copy()

    elif isinstance(file_or_df, (str, Path)):
        df = pd.read_csv(file_or_df)

    elif hasattr(file_or_df, "read"):  # Streamlit UploadedFile or file-like
        df = pd.read_csv(file_or_df)

    else:
        raise ValueError(f"Unsupported input type: {type(file_or_df)}")

    # ----------------------------------------------------------
    #  Basic Cleaning
    # ----------------------------------------------------------
    df.columns = df.columns.str.strip()  # remove whitespace
    df = df.drop_duplicates().reset_index(drop=True)

    # Handle TotalCharges (some blank or spaces)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # Handle tenure (ensure numeric)
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
        df["tenure"] = df["tenure"].clip(lower=0.01)  # avoid zero durations

    # ----------------------------------------------------------
    #  Encode Churn column
    # ----------------------------------------------------------
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0}).astype(int)
    else:
        df["Churn"] = 0  # assume retained if missing (for new customers)

    # ----------------------------------------------------------
    # Categorical Encoding
    # ----------------------------------------------------------
    # Identify categorical features (excluding numeric and survival cols)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # ----------------------------------------------------------
    #  Handle remaining missing values
    # ----------------------------------------------------------
    df = df.fillna(0)

    # ----------------------------------------------------------
    #  Final checks
    # ----------------------------------------------------------
    required_cols = ["tenure", "Churn"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


# ----------------------------------------------------------
# Run standalone (for debugging)
# ----------------------------------------------------------
if __name__ == "__main__":
    default_path = Path("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if default_path.exists():
        data = preprocess_data(default_path)
        print("✅ Preprocessing successful!")
        print(data.head())
    else:
        print("⚠️ Default dataset not found.")
