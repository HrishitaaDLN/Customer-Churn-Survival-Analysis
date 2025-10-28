import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
# Load dataset
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop(["customerID", "TotalCharges"], axis=1, inplace=True)
df["Churn"] = df["Churn"].map({"Yes": True, "No": False})

# One-hot encode categorical vars
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

# Split data
X = df.drop(["Churn", "tenure"], axis=1)
y = np.rec.fromarrays([df["Churn"].astype(bool), df["tenure"]], names=["event", "time"])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Survival Forest
rsf = RandomSurvivalForest(n_estimators=100, random_state=0)
rsf.fit(X_train, y_train)

# Evaluate
c_index = rsf.score(X_val, y_val)
print(f"✅ Concordance Index (validation): {c_index:.3f}")

# Save model
joblib.dump(rsf, "../app/model.pkl")
print("✅ Model saved to app/model.pkl")
