import streamlit as st
from predict import predict_heart_disease
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)
st.title("❤️ Heart Disease Prediction")
st.write(
    "This application uses a machine learning model "
    "to estimate the predicted risk of heart disease."
)
st.info(
    "Enter the patient's information to receive a prediction"
)

st.subheader("Patient Information")


with st.form("patient_form"):

    # Divide the form into two columns
    left_column, right_column = st.columns(2)

    with left_column:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=50,
            step=1
        )

        sex = st.selectbox(
            "Sex",
            options=["M", "F"]
        )

        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            options=["TA", "ATA", "NAP", "ASY"]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=120,
            step=1
        )

        cholesterol = st.number_input(
            "Cholesterol",
            min_value=0,
            max_value=700,
            value=200,
            step=1,
            help="Enter 0 if the cholesterol value is unknown."
        )

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1]
        )

    with right_column:

        resting_ecg = st.selectbox(
            "Resting ECG",
            options=["Normal", "ST", "LVH"]
        )

        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=40,
            max_value=250,
            value=150,
            step=1
        )

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            options=["Y", "N"]
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=-3.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        st_slope = st.selectbox(
            "ST Slope",
            options=["Up", "Flat", "Down"]
        )

    submitted = st.form_submit_button(
        "Predict Heart Disease Risk"
    )
if submitted:

    patient_data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain_type,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    try:
        with st.spinner("Analyzing patient information ..."):
            result = predict_heart_disease(patient_data=patient_data)
        prediction = result["prediction"]
        probability = result["probability"]
        threshold = result["threshold"]
        model_version = result["model_version"]

        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Heart Disease Probability",
            value=f" {probability:.1%} "
        )
        if prediction == 1:
            st.error(
                "The model detected a higher predicted risk "
                "of heart disease. Please consult a qualified "
                "healthcare professional."
            )
        else:
            st.success(
                "The model detected a lower predicted risk "
                "of heart disease."
            )
        st.caption(
            f"Decision threshold: {threshold:.2f} | "
            f"Model version: {model_version}"
        )
    except (ValueError, TypeError) as error:
        st.error(f"Invalid patient data: {error}")
st.warning(
    "This application is an educational machine learning project."
    "It does not provide a medical diagnosis"
)
