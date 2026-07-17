# Heart Disease Prediction and Deployment

An end-to-end machine learning project for heart disease risk prediction, covering data analysis, model tuning, threshold selection, reusable inference, automated testing, and deployment with Streamlit.

## Project Overview

This project predicts whether a patient has heart disease using machine learning.

It is a binary classification problem:

- `0`: No Heart Disease
- `1`: Heart Disease

The main focus was not only accuracy, but also recall and false negatives because missing a real heart disease patient is an important error.

---
## Live Application

The trained model is deployed as an interactive Streamlit web application.

**Live Demo:** [Open the Heart Disease Prediction App](https://raghad-heart-disease-prediction.streamlit.app/)

> This application is an educational machine learning project and does not provide a medical diagnosis.

---
## Dataset

The dataset contains:

- 918 rows
- 12 columns
- Medical features such as age, blood pressure, cholesterol, heart rate, chest pain type, ECG results, and exercise-related indicators

Target column:

- `HeartDisease`

---

## Main Steps

The project included:

- Exploratory Data Analysis
- Detecting hidden missing values
- Data cleaning
- Encoding categorical features
- Train-test split
- StandardScaler
- KNNImputer
- Baseline Logistic Regression
- Cross-validation
- Model comparison
- Gradient Boosting tuning with GridSearchCV
- XGBoost tuning with RandomizedSearchCV
- Overfitting and underfitting analysis
- False negative analysis
- Feature engineering experiment
- Feature importance

---

## Data Cleaning

Some values were unrealistic:

- `Cholesterol = 0`
- `RestingBP = 0`

The row with `RestingBP = 0` was removed.

Zero cholesterol values were replaced with missing values and handled using `KNNImputer`.

Preprocessing was done inside a pipeline to avoid data leakage.

---

## Models Tested

The following models were tested:

- Logistic Regression
- KNN
- SVM
- Random Forest
- Gradient Boosting
- XGBoost

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Recall
- Precision
- F1-score
- ROC-AUC
- Confusion Matrix
- False Negatives
- False Positives

Recall was the main metric because false negatives are especially important in medical classification.

---

## Final Model

The tuned Gradient Boosting pipeline was selected as the final model.

Its test results were approximately:

- Accuracy: 84.8%
- Recall: 96.1%
- Precision: 80.3%
- F1-score: 87.5%
- ROC-AUC: 93.8%

The training and testing results were close, which showed good generalization and reduced overfitting.

---
## Deployment Pipeline

The final application accepts the original 11 patient features directly.

The deployed workflow is:

1. Receive raw patient data from the Streamlit form.
2. Validate feature names, ranges, and categorical values.
3. Apply preprocessing using the saved scikit-learn pipeline.
4. Generate the heart disease probability.
5. Apply the selected classification threshold of `0.30`.
6. Display the prediction and probability to the user.

The preprocessing pipeline and trained Gradient Boosting model were saved together using `joblib`. This ensures that the application uses exactly the same transformations that were used during training.

---
## Decision Threshold

The final classification threshold is `0.30`.

This threshold was selected using out-of-fold probabilities generated from the training data with cross-validation. It was chosen to prioritize recall and reduce false negatives.

The test set was used only for final evaluation after the threshold had been selected.

The threshold is stored with the model artifact so that the notebook, prediction module, tests, and deployed application all use the same value.

---

## Prediction Testing

The deployment code includes automated tests for:

- Valid patient prediction
- Missing input features
- Invalid numerical values
- Invalid categorical values
- Unexpected input features

The tests can be run using:

```bash
python -m unittest test_predict.py
```

---

## XGBoost Experiment

XGBoost initially showed strong overfitting.

After tuning with `RandomizedSearchCV`, the train-test gap became much smaller.

The tuned XGBoost model achieved strong and balanced performance, but it did not outperform the tuned Gradient Boosting model overall.

---

## Error Analysis

False negative analysis showed that the model mainly missed heart disease patients with weaker risk signals, such as:

- Higher MaxHR
- No exercise-induced angina
- Lower Oldpeak values

Feature engineering was tested based on these patterns, but it did not improve the final model.

---

## Final Conclusion

The tuned Gradient Boosting pipeline was selected because it provided the best balance between:

- Recall
- F1-score
- ROC-AUC
- Generalization
- False negative reduction

This project helped me understand that a strong machine learning model is not selected by accuracy alone. Model stability, error analysis, overfitting, and the real-world goal are also important.

---

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Unittest
- Jupyter Notebook
- Git and GitHub

---

## Project Structure

| File | Purpose |
|---|---|
| `heart_disease_tuning.ipynb` | Data analysis, experimentation, model selection, and evaluation |
| `heart.csv` | Original dataset |
| `heart_disease_model.joblib` | Saved preprocessing pipeline, model, threshold, and metadata |
| `predict.py` | Input validation and reusable prediction logic |
| `test_predict.py` | Automated tests for the prediction module |
| `app.py` | Streamlit web application |
| `requirements.txt` | Exact Python dependencies required to run the project |
| `README.md` | Project documentation |
| `.gitignore` | Files and directories excluded from Git |

---

## Author

Raghad Abed
