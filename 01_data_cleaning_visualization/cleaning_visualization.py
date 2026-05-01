# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("First 5 rows:")
print(data.head())


# =========================
# DATA CLEANING
# =========================

# Remove duplicates
data = data.drop_duplicates()

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')

# Handle missing values
data = data.dropna()

print("\nCleaned Data Shape:", data.shape)


# =========================
# VISUALIZATION
# =========================

# 1. Churn Count
plt.figure()
sns.countplot(x='Churn', data=data)
plt.title("Customer Churn Count")
plt.show()

# 2. Monthly Charges vs Churn
plt.figure()
sns.boxplot(x='Churn', y='MonthlyCharges', data=data)
plt.title("Monthly Charges vs Churn")
plt.show()

# 3. Tenure vs Churn
plt.figure()
sns.boxplot(x='Churn', y='tenure', data=data)
plt.title("Tenure vs Churn")
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(10,6))
numeric_data = data.select_dtypes(include=['int64','float64'])
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
