from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline
model = joblib.load("model.pkl")

# SAME columns as training
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

categorical_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]


@app.route('/')
def home():
    return "Churn Prediction API Running 🚀"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # 🔥 EXACT SAME preprocessing as training

        # Fix TotalCharges
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

        # Convert numeric
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert categorical (VERY IMPORTANT)
        for col in categorical_cols:
            df[col] = df[col].astype(str)

        # Predict
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "churn_probability": round(float(prob), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)