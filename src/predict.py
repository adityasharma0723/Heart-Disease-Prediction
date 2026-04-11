import os
import joblib
import numpy as np
import pandas as pd


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def load_artifacts():
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    features_path = os.path.join(MODEL_DIR, "feature_names.joblib")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path, features_path]):
        raise FileNotFoundError(
            "Model artifacts not found. Run `python main.py` first to train and save the model."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    feature_names = joblib.load(features_path)

    return model, scaler, encoder, feature_names


def predict_heart_disease(patient_data):
    model, scaler, encoder, feature_names = load_artifacts()

    df = pd.DataFrame([patient_data])
    df_encoded = pd.get_dummies(df, drop_first=True)

    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_names]

    X_scaled = scaler.transform(df_encoded)

    prediction = model.predict(X_scaled)[0]
    prediction_label = encoder.inverse_transform([prediction])[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X_scaled)[0][1]
    else:
        probability = 0.0

    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.7:
        risk_level = "Medium"
    else:
        risk_level = "High"

    result = {
        "prediction": prediction_label,
        "probability": round(float(probability), 4),
        "risk_level": risk_level
    }

    return result


def save_model_artifacts(data, output_dir=None):
    if output_dir is None:
        output_dir = MODEL_DIR

    os.makedirs(output_dir, exist_ok=True)

    model = data.get("tuned_model", data.get("best_model"))
    model_name = data.get("best_model_name", "Unknown")

    joblib.dump(model, os.path.join(output_dir, "best_model.joblib"))
    joblib.dump(data["scaler"], os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(data["label_encoder"], os.path.join(output_dir, "label_encoder.joblib"))
    joblib.dump(data["feature_names"], os.path.join(output_dir, "feature_names.joblib"))

    print(f"\n[SAVED] Model artifacts saved to '{output_dir}/':")
    print(f"  - best_model.joblib    ({model_name})")
    print(f"  - scaler.joblib")
    print(f"  - label_encoder.joblib")
    print(f"  - feature_names.joblib")


if __name__ == "__main__":
    sample = {
        "Age": 55,
        "Gender": "Male",
        "Blood Pressure": 145,
        "Cholesterol Level": 260,
        "Exercise Habits": "Low",
        "Smoking": "Yes",
        "Family Heart Disease": "Yes",
        "Diabetes": "No",
        "BMI": 30.5,
        "High Blood Pressure": "Yes",
        "Low HDL Cholesterol": "Yes",
        "High LDL Cholesterol": "No",
        "Alcohol Consumption": "Medium",
        "Stress Level": "High",
        "Sleep Hours": 5.5,
        "Sugar Consumption": "High",
        "Triglyceride Level": 200,
        "Fasting Blood Sugar": 130,
        "CRP Level": 8.5,
        "Homocysteine Level": 15.0,
    }

    result = predict_heart_disease(sample)
    print(f"\n{'=' * 40}")
    print(f"  HEART DISEASE PREDICTION")
    print(f"{'=' * 40}")
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Probability : {result['probability']:.2%}")
    print(f"  Risk Level  : {result['risk_level']}")
