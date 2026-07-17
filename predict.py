from pathlib import Path
import joblib
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_disease_model.joblib"


def load_model_bundle():
    """Load the saved deployment bundle."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")
    model_bundle = joblib.load(MODEL_PATH)
    return model_bundle


# Load the model once when the file starts
bundle = load_model_bundle()

pipeline = bundle["pipeline"]
threshold = bundle["threshold"]
expected_features = bundle["features"]
# Run this test code only when predict.py is executed directly,
# not when it is imported into another file such as app.py.
# if __name__ == "__main__":
#    bundle = load_model_bundle()
#    print("Model loaded successfully.")
#    print("Threshold:", bundle["threshold"])
#    print("Model version:", bundle["model_version"])
#    print("Features:", bundle["features"])
ALLOWED_CATEGORIES = {
    "Sex": ["M", "F"],
    "ChestPainType": ["TA", "ATA", "NAP", "ASY"],
    "FastingBS": [0, 1],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["Y", "N"],
    "ST_Slope": ["Up", "Flat", "Down"]
}

NUMERIC_RANGES = {
    "Age": (18, 100),
    "RestingBP": (50, 250),
    "Cholesterol": (0, 700),
    "MaxHR": (40, 250),
    "Oldpeak": (-3, 10)
}


def validate_patient_data(patient_data):
    if not isinstance(patient_data, dict):
        raise TypeError("Patient data must be provided as a dictionary.")

    missing_features = [
        feature
        for feature in expected_features
        if feature not in patient_data
    ]
    if missing_features:
        raise ValueError(
            f" Missing required features: {missing_features}"
        )
    unexpected_features = [
        feature
        for feature in patient_data
        if feature not in expected_features
    ]
    if unexpected_features:
        raise ValueError(
            f"Unexpected features: {unexpected_features}"
        )
    for feature, allowed_values in ALLOWED_CATEGORIES.items():
        if patient_data[feature] not in allowed_values:
            raise ValueError(
                f"{feature} must be one of {allowed_values}."
            )
    for feature, (minimum, maximum) in NUMERIC_RANGES.items():
        value = patient_data[feature]
        if feature == "Cholesterol" and value in [0, None]:
            continue
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{feature} must be a number."
            )
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{feature} must be between "
                f"{minimum} and {maximum}."
            )
    normalized_data = patient_data.copy()
    if normalized_data["Cholesterol"] in [0, None]:
        normalized_data["Cholesterol"] = float("nan")
    return normalized_data


def predict_heart_disease(patient_data):
    validated_data = validate_patient_data(patient_data)

    patient_df = pd.DataFrame([validated_data], columns=expected_features)
    probability = pipeline.predict_proba(patient_df)[0, 1]
    prediction = int(probability >= threshold)
    return {
        "prediction": prediction,
        "probability": float(probability),
        "threshold": threshold,
        "model_version": bundle["model_version"]
    }


if __name__ == "__main__":
    test_patient = {
        "Age": 50,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 140,
        "Cholesterol": 220,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 150,
        "ExerciseAngina": "Y",
        "Oldpeak": 1.5,
        "ST_Slope": "Flat"
    }
    result = predict_heart_disease(test_patient)
    print(result)
