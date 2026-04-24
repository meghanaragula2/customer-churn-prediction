import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID
data.drop("customerID", axis=1, inplace=True)

# 🔥 FIX 1: Convert TotalCharges
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(0)

# 🔥 FIX 2: Convert SeniorCitizen to STRING (VERY IMPORTANT)
data['SeniorCitizen'] = data['SeniorCitizen'].astype(str)

# Target variable
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Split
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Column groups
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

categorical_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "model.pkl")

print("✅ Model trained and saved!")
