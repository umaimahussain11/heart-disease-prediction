# ❤️ Heart Disease Risk Prediction

A beginner-friendly machine learning project that predicts the likelihood of heart disease using patient health information.

## 🚀 Live Demo

Try the deployed application here:

👉 https://heart-disease-prediction-bkt2gcmjgxppcx7avkytwk.streamlit.app/

## 📌 Project Overview

This project uses machine learning to classify whether a patient is likely to belong to a lower-risk or higher-risk heart disease category based on medical attributes.

The project includes:

- Data preprocessing
- Missing value handling
- Exploratory analysis
- Machine learning model training
- Logistic Regression
- Random Forest Classifier
- Model evaluation
- Confusion matrix
- Classification report
- Streamlit web application

## 🧠 Machine Learning Models

Two classification models were tested:

### Logistic Regression
Accuracy: **86.89%**

### Random Forest
Accuracy: **88.52%**

The Random Forest model performed slightly better and was therefore saved for use in the Streamlit application.

## 📊 Dataset

The dataset contains **303 records** and **13 input features**.

Features include:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- ST depression
- Slope
- Number of major vessels
- Thalassemia

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

## 🌳 Random Forest Performance

Accuracy: **88.52%**

Classification performance:

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| 0 | 0.93 | 0.85 | 0.89 |
| 1 | 0.84 | 0.93 | 0.88 |

## 🌐 Streamlit Application

The trained Random Forest model is integrated into a Streamlit web application.

Users can enter patient information and receive a machine-learning prediction.

The application is intended for **educational purposes only** and should not be used as a medical diagnosis tool.

## 📁 Project Structure

```text
heart-disease-prediction/
│
├── app.py
├── main.py
├── heart_disease_model.pkl
├── requirements.txt
├── README.md
└── .gitignore

## 📊 Model Performance

Two machine learning classification models were evaluated:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 86.89% |
| Random Forest | 88.52% |

### Random Forest Classification Report

| Metric | Class 0 | Class 1 |
|---|---:|---:|
| Precision | 0.93 | 0.84 |
| Recall | 0.85 | 0.93 |
| F1-Score | 0.89 | 0.88 |

### Confusion Matrix

```text
[[28  5]
 [ 2 26]]