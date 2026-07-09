# Heart Disease Prediction with Model Tuning

## Project Overview

This project predicts whether a patient has heart disease using machine learning.

It is a binary classification problem:

- `0`: No Heart Disease
- `1`: Heart Disease

The main focus was not only accuracy, but also recall and false negatives because missing a real heart disease patient is an important error.

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
- Jupyter Notebook

---

## Files

- `heart.csv`
- `heart_disease_tuning.ipynb`
- `README.md`
- `.gitignore`

---

## Author

Raghad Abed
