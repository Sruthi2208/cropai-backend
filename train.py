import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Ensure artifacts folder exists
os.makedirs("artifacts", exist_ok=True)

# Load dataset
df = pd.read_csv("../dataset/crop_recommendation.csv")

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & label encoder
joblib.dump(model, "artifacts/model.joblib")
joblib.dump(le, "artifacts/label_encoder.joblib")

print("✅ Model trained and saved in artifacts/")
