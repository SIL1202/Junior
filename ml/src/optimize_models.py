import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
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
models = {
    'Logistic Regression (SMOTE)': LogisticRegression(max_iter=1000, random_state=42),
    'Logistic Regression (Weighted)': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest (SMOTE)': RandomForestClassifier(n_estimators=100, random_state=42),
    'Random Forest (Weighted)': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'XGBoost (Weighted)': XGBClassifier(scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), eval_metric='logloss', random_state=42)
}

results = []

for name, model in models.items():
    print(f"Training {name}...")
    if 'SMOTE' in name:
        model.fit(X_train_smote, y_train_smote)
    else:
        model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, f'models/{name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.joblib')
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
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
    
    print(f"Recall for {name}: {metrics['Recall']:.4f}")

# Comparison Table
optimized_df = pd.DataFrame(results)
print("\n--- Optimized Model Comparison Table ---")
print(optimized_df)
optimized_df.to_csv('reports/optimized_model_comparison.csv', index=False)

print("\nOptimization complete. Check reports/optimized_model_comparison.csv for details.")
