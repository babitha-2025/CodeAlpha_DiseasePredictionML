# Heart Disease Prediction using Machine Learning

## Project Overview

This project predicts the presence of heart disease using patient medical data and machine learning techniques.

The project uses the UCI Heart Disease dataset for binary heart disease prediction.

## Dataset

The dataset used in this project is the UCI Heart Disease Dataset.

The original target variable contains values from 0 to 4:

- 0 = No heart disease
- 1-4 = Presence of heart disease

For this project, the target was converted into a binary classification:

- 0 = No Disease
- 1 = Disease

The dataset contains 303 patient records and 13 input features.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib

## Machine Learning Model

The main model used in this project is:

**XGBoost Classifier**

The model was trained using an 80/20 train-test split with stratification.

## Model Performance

The XGBoost model achieved:

**Accuracy: 91.8%**

### Confusion Matrix

```text
[[29  4]
 [ 1 27]]