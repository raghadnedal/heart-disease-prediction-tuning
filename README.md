# Heart Disease Prediction with Model Tuning

## Overview

This project is a binary classification machine learning project that predicts whether a patient has heart disease or not.

The goal of this project was not only to build a model with good accuracy, but also to evaluate the model correctly, reduce false negatives, and understand where the model makes mistakes.

Since this is a medical classification problem, the most important metric is Recall. A false negative means that a patient who actually has heart disease is predicted as healthy, which can be dangerous.

---

## Dataset

The dataset used in this project is the Heart Failure Prediction Dataset from Kaggle.

The dataset contains:

- 918 rows
- 12 columns
- Binary target column: HeartDisease

Target values:

- 0: No Heart Disease
- 1: Heart Disease

---

## Problem Type

This is a binary classification problem.

The model predicts one of two classes:

- No Heart Disease
- Heart Disease

---

## Features

The dataset contains the following features:

- Age: Patient age in years
- Sex: Patient gender
- ChestPainType: Type of chest pain
- RestingBP: Resting blood pressure
- Cholesterol: Serum cholesterol level
- FastingBS: Fasting blood sugar
- RestingECG: Resting ECG result
- MaxHR: Maximum heart rate achieved
- ExerciseAngina: Exercise-induced angina
- Oldpeak: ST depression caused by exercise compared to rest
- ST_Slope: Slope of the ST segment
- HeartDisease: Target column

---

## Project Workflow

This project followed a full machine learning workflow:

1. Understanding the problem
2. Loading the dataset
3. Exploratory Data Analysis
4. Understanding the features and target
5. Detecting hidden missing values
6. Data cleaning
7. Encoding categorical features
8. Train-test split
9. Scaling and KNN imputation
10. Baseline model training
11. Model evaluation
12. Cross-validation
13. Model comparison
14. Pipeline-based evaluation
15. Threshold tuning
16. Error analysis
17. Feature engineering experiment
18. Feature importance
19. Final model selection

---

## Exploratory Data Analysis

During EDA, I checked:

- Dataset shape
- Column types
- Missing values
- Target distribution
- Basic statistics
- Unrealistic medical values

The target column was fairly balanced:

- Heart Disease: around 55%
- No Heart Disease: around 45%

This means the dataset was not severely imbalanced.

---

## Data Cleaning

The dataset did not contain normal missing values as NaN, but some medical columns had unrealistic zero values.

I found:

- Cholesterol = 0 in many rows
- RestingBP = 0 in one row

These values are medically unrealistic, so I treated them as hidden missing values.

Cleaning decisions:

- The row with RestingBP = 0 was removed because it appeared only once.
- Cholesterol = 0 was replaced with NaN because many rows had this issue.

To avoid data leakage, I did not apply imputation directly on the full dataset. Instead, imputation was handled later inside a machine learning pipeline.

---

## Preprocessing

The preprocessing steps included:

- Encoding categorical columns
- Scaling numerical features
- KNN imputation for missing Cholesterol values

I used KNNImputer because it fills missing values based on similar patients.

Since KNNImputer uses distances between rows, I used StandardScaler before imputation.

The final preprocessing pipeline included:

- StandardScaler
- KNNImputer

---

## Models Used

Several classification models were tested and compared:

- Logistic Regression
- SVM
- KNN
- Random Forest
- Gradient Boosting

The models were compared using 5-fold cross-validation.

---

## Evaluation Metrics

The main evaluation metrics were:

- Recall
- Precision
- F1-score
- ROC-AUC
- Confusion Matrix
- Accuracy

Accuracy was reported, but it was not the main metric.

Since this is a medical classification problem, Recall was the most important metric.

Recall answers this question:

Out of all patients who actually have heart disease, how many did the model correctly detect?

A false negative is the most dangerous error in this project because it means a sick patient is predicted as healthy.

---

## Baseline Model

The baseline model was Logistic Regression.

It performed well, but after using cross-validation, the results showed that its performance was not the best compared to other models.

This step was important because the baseline model gave a starting point for comparison.

---

## Cross-Validation

I used 5-fold cross-validation to get a more reliable estimate of model performance.

Instead of relying on one train-test split, cross-validation evaluates the model on multiple different splits.

This helped show whether the model performance was stable or dependent on one specific split.

---

## Model Comparison

After comparing multiple models using cross-validation, Gradient Boosting achieved the best recall.

Model comparison showed that:

- Gradient Boosting had the best mean recall.
- Random Forest had a very strong ROC-AUC.
- Logistic Regression was a good baseline but not the best final model.

Because recall is the most important metric in this medical problem, Gradient Boosting was selected as the best candidate.

---

## Final Pipeline

The final model was a Gradient Boosting Classifier inside a full preprocessing pipeline.

The pipeline included:

- StandardScaler
- KNNImputer
- GradientBoostingClassifier

Using a pipeline helped avoid data leakage because preprocessing was fitted only on the training data during cross-validation and final training.

---

## Threshold Tuning

The default classification threshold is 0.50.

However, in medical classification, the default threshold is not always the best choice.

I tested different thresholds:

- 0.30
- 0.35
- 0.40
- 0.45
- 0.50

The goal was to improve recall and reduce false negatives.

The selected threshold was 0.30.

This threshold reduced the number of missed heart disease patients compared to the default threshold.

---

## Final Results

The final selected model was:

Gradient Boosting Pipeline with threshold = 0.30

Final results:

- Accuracy: 86%
- Recall: 91.2%
- Precision: 85%
- F1-score: 88%
- False Negatives: 9
- False Positives: 17

The model achieved high recall, which is important because the goal was to detect as many actual heart disease patients as possible.

---

## Confusion Matrix Interpretation

The final confusion matrix showed:

- True Negatives: healthy patients correctly predicted as healthy
- False Positives: healthy patients predicted as having heart disease
- False Negatives: heart disease patients predicted as healthy
- True Positives: heart disease patients correctly detected

The most important number was False Negatives.

After threshold tuning, the model reduced false negatives to 9.

This means the model missed fewer heart disease patients.

---

## Error Analysis

After selecting the threshold of 0.30, I analyzed the false negative cases.

False negatives are patients who actually had heart disease but were predicted as healthy.

The analysis showed that the model mainly missed patients with weaker risk signals, such as:

- Higher MaxHR
- No exercise-induced angina
- Lower Oldpeak
- Lower predicted probability

This suggests that the model struggles with heart disease patients who do not show strong typical risk patterns.

This was an important step because it helped explain where the model fails instead of only reporting performance metrics.

---

## Feature Engineering Experiment

Based on the false negative analysis, I created new features:

- High_MaxHR
- Low_Oldpeak
- No_ExerciseAngina
- Silent_Risk

The goal was to help the model detect patients with weaker heart disease signals.

However, after testing the model again, feature engineering did not improve recall or reduce false negatives.

Before feature engineering:

- False Negatives: 9
- False Positives: 17

After feature engineering:

- False Negatives: 9
- False Positives: 18

Because the new features did not improve the model, I decided not to use the feature-engineered version as the final model.

The simpler Gradient Boosting pipeline with threshold tuning was selected.

---

## Feature Importance

Feature importance was used to understand which variables the Gradient Boosting model relied on the most.

The most important features included:

- ST_Slope
- Oldpeak
- MaxHR
- Sex
- Cholesterol
- RestingBP

The model relied mostly on ECG-related and exercise-related features.

This makes sense because ECG and exercise-related indicators are strongly related to heart disease risk.

Feature importance helped make the model more interpretable.

---

## Final Conclusion

The final model was a Gradient Boosting Classifier inside a preprocessing pipeline with a selected threshold of 0.30.

This model was selected because it reduced false negatives and achieved strong recall, which is the most important metric in this medical classification problem.

The project showed that building a good machine learning model is not only about getting high accuracy.

It is also about:

- Choosing the right evaluation metric
- Avoiding data leakage
- Comparing models correctly
- Tuning the classification threshold
- Analyzing errors
- Understanding where the model fails
- Making decisions based on the real problem context

---

## What I Learned

In this project, I learned:

- How to build a binary classification project
- How to identify features and target
- How to perform EDA on a medical dataset
- How to detect hidden missing values
- Why unrealistic zero values can represent missing data
- How to avoid data leakage
- How to use KNNImputer
- Why scaling is important before KNN-based methods
- How to use pipelines professionally
- How to compare models using cross-validation
- Why recall is more important than accuracy in medical problems
- How threshold tuning can reduce false negatives
- How to analyze false negative cases
- How to test feature engineering ideas
- Why more features do not always improve the model
- How to interpret a tree-based model using feature importance

---

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Project Structure

heart-disease-prediction-tuning/

- heart.csv
- heart_disease_tuning.ipynb
- README.md
- .gitignore

---

## Author

Raghad Abed
