import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Ensure output directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

# Load processed data
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Results storage
results = []

# Evaluation and Plotting setup
plt.figure(figsize=(10, 8))

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, f'models/{name.lower().replace(" ", "_")}.joblib')
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    # Metrics
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC': roc_auc
    }
    results.append(metrics)
    
    # Print Classification Report
    print(f"\n--- {name} Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"--- {name} Confusion Matrix ---")
    print(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'reports/figures/confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

# Finalize ROC Plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right')
plt.savefig('reports/figures/roc_curves.png')
plt.close()

# Comparison Table
comparison_df = pd.DataFrame(results)
print("\n--- Model Comparison Table ---")
print(comparison_df)
comparison_df.to_csv('reports/model_comparison.csv', index=False)

# 7. Feature Importance (for Random Forest and XGBoost)
def plot_importance(model, name, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10
    plt.figure(figsize=(10, 6))
    plt.title(f'Top 10 Feature Importances - {name}')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(f'reports/figures/feature_importance_{name.lower().replace(" ", "_")}.png')
    plt.close()

plot_importance(models['Random Forest'], 'Random Forest', X_train.columns)
plot_importance(models['XGBoost'], 'XGBoost', X_train.columns)

print("\nModel training and evaluation complete. Results saved to reports/")
