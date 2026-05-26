import pandas as pd
import os

# Load results
initial_results = pd.read_csv('reports/model_comparison.csv')
optimized_results = pd.read_csv('reports/optimized_model_comparison.csv')

# Combine and highlight the best performer
all_results = pd.concat([initial_results, optimized_results], ignore_index=True)
all_results = all_results.sort_values(by='Recall', ascending=False)

print("--- FINAL MODEL PERFORMANCE SUMMARY ---")
print(all_results[['Model', 'Accuracy', 'Recall', 'AUC']].to_string(index=False))

best_model = all_results.iloc[0]
print(f"\n🏆 Best Model for Recall: {best_model['Model']}")
print(f"   Recall: {best_model['Recall']:.2%}")
print(f"   Accuracy: {best_model['Accuracy']:.2%}")

# Save final report
all_results.to_csv('reports/final_summary.csv', index=False)

print("\nAll project milestones completed:")
print("✅ Week 14: Data Profiling & EDA")
print("✅ Week 15: Feature Engineering & Model Training")
print("✅ Week 16: Model Optimization & Final Evaluation")
print("\nOutput folders:")
print("- reports/figures/: EDA and Model Performance plots")
print("- models/: Saved model files (.joblib)")
print("- data/processed/: Prepared datasets")
