# HEART DISEASE PREDICTION COMPLETE CODE

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Add file upload here

# ==========================
# LOAD DATASET
# ==========================
df = pd.read_csv("heart_disease_dataset.csv")
print(df.columns)
print(df.shape)
df.head()

# Clean Data
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

df = df.drop_duplicates()

print(df.shape)

#Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(df)
plt.show()
# Target Distribution
sns.countplot(x='heart_disease', data=df)
plt.title("Heart Disease Distribution")
plt.show()

# Correlation Heatmap

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Age Distribution

sns.histplot(df['age'], kde=True)
plt.show()

# Cholesterol vs Heart Disease

sns.boxplot(x='heart_disease', y='cholesterol', data=df)
plt.show()
# 6. Preprocessing
from sklearn.preprocessing import StandardScaler

X = df.drop('heart_disease', axis=1)
y = df['heart_disease']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Remove duplicates
df = df.drop_duplicates()

# ==========================
# FEATURE ENGINEERING
# ==========================
df['bp_risk'] = (df['resting_blood_pressure'] > 140).astype(int)
df['chol_risk'] = (df['cholesterol'] > 240).astype(int)
df['hr_ratio'] = df['max_heart_rate'] / df['age']

# ==========================
# INPUT / OUTPUT
# ==========================
X = df.drop('heart_disease', axis=1)
y = df['heart_disease']


# Save feature names
feature_names = X.columns.tolist()

# ==========================
# SCALING
# ==========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================
# MODEL TRAINING
# ==========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, pred))

# Save model and scaler
joblib.dump(model, "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")

print("\nModel Saved Successfully!")

# ==========================
# USER INPUT SECTION
# ==========================
print("\nEnter Patient Details:")

age = float(input("Enter Age: "))
sex = int(input("Enter Sex (1=Male, 0=Female): "))
cp = int(input("Chest Pain Type (0-3): "))
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Cholesterol: "))
fbs = int(input("Fasting Blood Sugar >120 (1=True,0=False): "))
restecg = int(input("Rest ECG (0-2): ")) # Corrected input variable name for consistency
thalach = float(input("Max Heart Rate: "))
exang = int(float(input("Exercise Angina (1=Yes,0=No): "))) # Modified to handle float inputs (e.g., 0.0 or 1.0) and truncate others
oldpeak = float(input("ST Depression: "))
slope = int(input("Slope (0-2): ")) # Corrected input variable name for consistency
ca = int(input("Number of Major Vessels (0-3): "))
thal = int(input("Thalassemia (0-3): "))

# Engineered Features
bp_risk = 1 if trestbps > 140 else 0
chol_risk = 1 if chol > 240 else 0
hr_ratio = thalach / age

# Create DataFrame (IMPORTANT: Corrected column names to match training features)
user_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'chest_pain_type': cp,
    'resting_blood_pressure': trestbps,
    'cholesterol': chol,
    'fasting_blood_sugar': fbs,
    'resting_ecg': restecg, # Changed 'rest_ecg' to 'resting_ecg'
    'max_heart_rate': thalach,
    'exercise_induced_angina': exang,
    'st_depression': oldpeak,
    'st_slope': slope, # Changed 'slope' to 'st_slope'
    'num_major_vessels': ca,
    'thalassemia': thal,
    'bp_risk': bp_risk,
    'chol_risk': chol_risk,
    'hr_ratio': hr_ratio
}])

# Load saved scaler/model
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# Scale input
user_scaled = scaler.transform(user_data)

# Predict
prediction = model.predict(user_scaled)

# Output
print("\nPrediction Result:")
if prediction[0] == 1:
    print("💀HIGH RISK: Heart Disease Detected")
else:
    print("✅ LOW RISK: No Heart Disease")