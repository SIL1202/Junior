import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/Telco Customer Churn.csv')

# 1. Basic Info
print("--- Dataset Info ---")
print(df.info())

# 2. Check for Missing Values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 3. Investigate TotalCharges (as mentioned in GEMINI.md)
print("\n--- TotalCharges Investigation ---")
# Check the type of TotalCharges
print(f"TotalCharges current dtype: {df['TotalCharges'].dtype}")

# Try to convert TotalCharges to numeric to find anomalies (e.g., empty strings)
df['TotalCharges_Numeric'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
missing_total_charges = df[df['TotalCharges_Numeric'].isnull()]
print(f"Number of rows where TotalCharges is non-numeric: {len(missing_total_charges)}")
if len(missing_total_charges) > 0:
    print("Example rows with non-numeric TotalCharges:")
    print(missing_total_charges[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']].head())

# 4. Summary Statistics for Numeric Features
print("\n--- Summary Statistics (Numeric) ---")
print(df.describe())

# 5. Class Balance (Churn)
print("\n--- Churn Class Balance ---")
print(df['Churn'].value_counts(normalize=True))
