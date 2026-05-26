import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output directory exists
os.makedirs('reports/figures', exist_ok=True)

# Load dataset
df = pd.read_csv('data/Telco Customer Churn.csv')

# --- Preprocessing for EDA ---
# Handle TotalCharges missing values (empty strings for tenure=0)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Set plotting style
sns.set_theme(style="whitegrid")

# 1. Churn Distribution (Pie Chart)
plt.figure(figsize=(6, 6))
df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99'])
plt.title('Churn Distribution')
plt.savefig('reports/figures/churn_distribution.png')
plt.close()

# 2. Tenure vs Churn
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, common_norm=False, palette='viridis')
plt.title('Tenure Distribution by Churn')
plt.savefig('reports/figures/tenure_vs_churn.png')
plt.close()

# 3. Monthly Charges vs Churn
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, palette='magma')
plt.title('Monthly Charges Distribution by Churn')
plt.savefig('reports/figures/monthly_charges_vs_churn.png')
plt.close()

# 4. Contract Type vs Churn
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Contract', hue='Churn', palette='muted')
plt.title('Churn by Contract Type')
plt.savefig('reports/figures/contract_vs_churn.png')
plt.close()

# 5. Internet Service vs Churn
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='InternetService', hue='Churn', palette='pastel')
plt.title('Churn by Internet Service')
plt.savefig('reports/figures/internet_service_vs_churn.png')
plt.close()

# 6. Correlation Heatmap (for numeric features)
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('reports/figures/correlation_heatmap.png')
plt.close()

print("EDA visualizations saved to reports/figures/")
