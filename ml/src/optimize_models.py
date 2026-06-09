import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Ensure output directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Load processed data
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# 1. SMOTE (Oversampling the minority class)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize optimized models
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
models = {
    'Logistic Regression (SMOTE)': LogisticRegression(max_iter=1000, random_state=42),
    'Logistic Regression (Weighted)': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest (SMOTE)': RandomForestClassifier(n_estimators=100, random_state=42),
    'Random Forest (Weighted)': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'XGBoost (Weighted)': XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.01,
        max_depth=3,
        n_estimators=50,
        subsample=1.0,
        eval_metric='logloss',
        random_state=42
    )
}

results = []

# Decision boundary threshold for optimized models
THRESHOLD = 0.45
print(f"Applying classification threshold of {THRESHOLD} to predictions...")

for name, model in models.items():
    print(f"Training {name}...")
    if 'SMOTE' in name:
        model.fit(X_train_smote, y_train_smote)
    else:
        model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, f'models/{name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.joblib')
    
    # Predictions (using custom decision threshold)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)
    
    # Metrics
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }
    results.append(metrics)
    
    print(f"\n--- {name} Classification Report (Threshold: {THRESHOLD}) ---")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"--- {name} Confusion Matrix ---")
    print(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
    plt.title(f'Confusion Matrix - {name}\n(Threshold: {THRESHOLD})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'reports/figures/confusion_matrix_{name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png')
    plt.close()

# Comparison Table
optimized_df = pd.DataFrame(results)
print("\n--- Optimized Model Comparison Table ---")
print(optimized_df)
optimized_df.to_csv('reports/optimized_model_comparison.csv', index=False)

print("\nOptimization complete. Check reports/optimized_model_comparison.csv for details.")
