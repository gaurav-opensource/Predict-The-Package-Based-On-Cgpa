import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import joblib

# Load dataset
dataset = pd.read_csv("cgpadata.csv")  # Keep CSV in same folder

# Define features and target
X = dataset[["CGPA"]]
y = dataset["Package_LPA"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=30)

# Train model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict and evaluate
y_pred = lr.predict(x_test)
accuracy = r2_score(y_test, y_pred)
print(f"Model Accuracy (R² score): {accuracy:.2f}")

# Save the trained model
joblib.dump(lr, "model.pkl")
print("Model saved as model.pkl")
