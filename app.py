from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ===============================
# Absolute path setup (VERY IMPORTANT)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# ===============================
# Load trained artifacts
# ===============================
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
kmeans = joblib.load(os.path.join(ARTIFACTS_DIR, "kmeans_model.pkl"))
churn_model = joblib.load(os.path.join(ARTIFACTS_DIR, "churn_model.pkl"))

THRESHOLD = 0.35   # tuned threshold for churn

# ===============================
# Routes
# ===============================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # --------- Read form inputs ---------
    age = int(request.form["age"])
    gender = int(request.form["gender"])
    income = float(request.form["income"])
    spending = float(request.form["spending"])
    membership = int(request.form["membership"])
    frequency = int(request.form["frequency"])

    # --------- SEGMENTATION ---------
    seg_df = pd.DataFrame(
        [[income, spending, frequency]],
        columns=[
            "Annual_Income",
            "Spending_Score",
            "Purchase_Frequency"
        ]
    )

    seg_scaled = scaler.transform(seg_df)
    segment = int(kmeans.predict(seg_scaled)[0])

    # --------- CHURN PREDICTION ---------
    churn_df = pd.DataFrame(
        [[age, gender, income, spending, membership, frequency, segment]],
        columns=[
            "Age",
            "Gender",
            "Annual_Income",
            "Spending_Score",
            "Membership_Level",
            "Purchase_Frequency",
            "segment"   # MUST be lowercase (training-time feature)
        ]
    )

    prob = churn_model.predict_proba(churn_df)[0][1]
    churn = "High Risk" if prob >= THRESHOLD else "Low Risk"

    # --------- Return result ---------
    return render_template(
        "index.html",
        segment=segment,
        churn=churn,
        probability=round(prob, 2)
    )


if __name__ == "__main__":
    app.run()
