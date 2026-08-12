from ucimlrepo import fetch_ucirepo

# Fetch the UCI Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

# Store features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine features and target into one DataFrame
import pandas as pd

df = pd.concat([X, y], axis=1)

print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nTarget distribution:")
print(df["num"].value_counts())

print("\nFirst 5 rows:")
print(df.head())

# Handle missing values
df["ca"] = df["ca"].fillna(df["ca"].median())
df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

# Convert target into binary classification
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

print("\nAfter cleaning:")
print("Missing values:")
print(df.isnull().sum())

print("\nNew target distribution:")
print(df["num"].value_counts())

# Separate features and target
X = df.drop("num", axis=1)
y = df["num"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

# Scale the features and train a Logistic Regression model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

print("\nModel training completed!")

# Make predictions on the test data
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# Exploratory Data Analysis
# ==============================

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Heart disease distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="num", data=df)
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.show()

# 2. Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", hue="num", bins=20, kde=True)
plt.title("Age Distribution by Heart Disease")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()

# 3. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# ==============================
# Random Forest Model
# ==============================

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

# Make predictions
rf_pred = rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:")
print(f"{rf_accuracy * 100:.2f}%")

# Evaluate Random Forest
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# ==============================
# Save the trained model
# ==============================

import joblib

joblib.dump(rf_model, "heart_disease_model.pkl")

print("\nRandom Forest model saved successfully!")

# Feature Importance
import matplotlib.pyplot as plt

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

feature_importance = feature_importance.sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind="barh")

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.tight_layout()
plt.show()