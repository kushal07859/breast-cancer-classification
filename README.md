
# Breast Cancer Classification Using Machine Learning

## Overview
This educational machine learning project uses the Breast Cancer Wisconsin (Diagnostic) dataset to classify samples as benign (B) or malignant (M).

## Objective
Build and evaluate a supervised classification model using Python and scikit-learn.

## Workflow
- Load and inspect the dataset
- Separate input features and target labels
- Split data into training and testing sets
- Scale features using `StandardScaler`
- Compare classification algorithms using cross-validation
- Tune Logistic Regression using `GridSearchCV`
- Evaluate the model using accuracy, precision, recall, F1-score, a confusion matrix, and ROC-AUC
- Save and reload the trained pipeline using Joblib

## Model
The selected model is a tuned Logistic Regression pipeline containing feature scaling and the classifier.

## Test Results
- Accuracy: approximately 97%
- ROC-AUC: approximately 0.996
- Test samples: 114

These are results from this particular test split and are not a guarantee of future performance.

## Technologies
Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib, and KaggleHub.

## Limitations
This is an educational project, not a clinically validated diagnostic tool. Its results should not be used to make medical decisions.

## How to Run
1. Install the required packages from `requirements.txt`.
2. Obtain the dataset from its original source.
3. Run the training script.
4. Run `predict.py` to load the saved model and make a prediction using correctly formatted input features.

