import unittest

from predict import predict_heart_disease


# Valid patient data used as a base for the tests
VALID_PATIENT = {
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


class TestHeartDiseasePrediction(unittest.TestCase):

    # Test that valid data returns a valid prediction
    def test_valid_patient_returns_prediction(self):

        result = predict_heart_disease(VALID_PATIENT)

        self.assertIn(result["prediction"], [0, 1])
        self.assertGreaterEqual(result["probability"], 0)
        self.assertLessEqual(result["probability"], 1)

    # Test that a missing feature is rejected
    def test_missing_feature_raises_error(self):

        patient = VALID_PATIENT.copy()
        patient.pop("Age")

        with self.assertRaises(ValueError):
            predict_heart_disease(patient)

    # Test that an invalid age is rejected
    def test_invalid_age_raises_error(self):

        patient = VALID_PATIENT.copy()
        patient["Age"] = -10

        with self.assertRaises(ValueError):
            predict_heart_disease(patient)

    # Test that an invalid category is rejected
    def test_invalid_category_raises_error(self):

        patient = VALID_PATIENT.copy()
        patient["Sex"] = "Unknown"

        with self.assertRaises(ValueError):
            predict_heart_disease(patient)

    # Test that zero cholesterol is treated as missing
    def test_zero_cholesterol_is_accepted_as_missing(self):

        patient = VALID_PATIENT.copy()
        patient["Cholesterol"] = 0

        result = predict_heart_disease(patient)

        self.assertIn(result["prediction"], [0, 1])
        self.assertGreaterEqual(result["probability"], 0)
        self.assertLessEqual(result["probability"], 1)


# Run the tests only when this file is executed directly
if __name__ == "__main__":
    unittest.main()
