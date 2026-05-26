import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv('data/Telco Customer Churn.csv')

# 1. Calculate 'Service Density'
# We count how many additional services a customer has 'Yes' for.
service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

def count_services(row):
    count = 0
    for col in service_cols:
        if row[col] == 'Yes':
            count += 1
    return count

df['ServiceDensity'] = df.apply(count_services, axis=1)

# 2. Analysis: Tenure vs Churn
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x='Churn', y='tenure', data=df, palette='Set2')
plt.title('Tenure vs Churn')
plt.xlabel('Churn (Yes/No)')
plt.ylabel('Tenure (Months)')

# 3. Analysis: Service Density vs Churn
plt.subplot(1, 2, 2)
sns.barplot(x='ServiceDensity', y='Churn', data=df.assign(Churn=df['Churn'].map({'Yes': 1, 'No': 0})), palette='viridis')
plt.title('Churn Rate by Service Density')
plt.xlabel('Number of Services (Service Density)')
plt.ylabel('Churn Rate')

plt.tight_layout()
os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/tenure_density_analysis.png')
plt.close()

# 4. Correlation Calculation
df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
tenure_corr = df['tenure'].corr(df['Churn_Numeric'])
density_corr = df['ServiceDensity'].corr(df['Churn_Numeric'])

print(f"Correlation between Tenure and Churn: {tenure_corr:.4f}")
print(f"Correlation between Service Density and Churn: {density_corr:.4f}")

# Grouped analysis
print("\n--- Churn Rate by Service Density Groups ---")
print(df.groupby('ServiceDensity')['Churn_Numeric'].mean())

print("\n--- Tenure Statistics by Churn ---")
print(df.groupby('Churn')['tenure'].describe())
