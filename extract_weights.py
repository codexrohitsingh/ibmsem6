import joblib
import pandas as pd
import numpy as np

# Load data and encoders
df = pd.read_csv("Placement_Data_Full_Class.csv")
df.drop(['salary', 'sl_no'], axis=1, inplace=True)

# Load model data
model_data = joblib.load('model_new.joblib')
lr = model_data['model']
encoders = model_data['encoders']
features = model_data['features']

# Calculate means for numerical columns and modes for categorical (encoded) columns
defaults = {}
for col in features:
    if col in encoders:
        # Categorical column - get mode and encode it
        mode_val = df[col].mode()[0]
        encoded_val = encoders[col].transform([mode_val])[0]
        defaults[col] = int(encoded_val)
    else:
        # Numerical column - get mean
        defaults[col] = float(df[col].mean())

print("--- MODEL WEIGHTS ---")
print(f"Intercept: {lr.intercept_[0]}")
print("Coefficients:")
for feat, coef in zip(features, lr.coef_[0]):
    print(f"  {feat}: {coef}")

print("\n--- DEFAULT VALUES ---")
for feat, val in defaults.items():
    print(f"  {feat}: {val}")

print("\n--- ENCODERS ---")
for col, le in encoders.items():
    print(f"  {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")
