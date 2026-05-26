# Telco Customer Churn Prediction and Model Comparison

## Project Overview
This project focuses on predicting customer churn for a telecommunications service provider. In subscription-based industries, retaining existing customers is significantly more cost-effective than acquiring new ones. This system implements machine learning models to identify high-risk customers, enabling proactive retention strategies.

## Performance Targets
The primary goal of this project was to prioritize Recall over overall Accuracy.
* Target Recall: Higher than 75%
* Achieved Recall: 78.34% (Logistic Regression - Weighted)
* Achieved Accuracy: 73.81%
* AUC-ROC: 0.84

## Repository Structure
* data/: Contains the raw dataset and processed training/testing sets.
* src/: Source code for data profiling, exploratory data analysis (EDA), feature engineering, and model training.
* models/: Saved model files (.joblib) and data scalers.
* reports/: Comparison tables and visualizations including ROC curves and feature importance plots.
* gemini.md: Detailed project report and findings.

## Methodology
1. Data Profiling: Initial analysis to handle missing values and understand data distribution.
2. EDA: Visualizing relationships between features like contract type, tenure, and churn.
3. Feature Engineering: Implementing one-hot encoding for categorical variables and standard scaling for numeric features.
4. Model Training: Comparing Logistic Regression, Random Forest, and XGBoost.
5. Optimization: Utilizing class weighting and SMOTE to address class imbalance and maximize recall.

## Key Insights
* Tenure: Customers in their first 12 months are at the highest risk of churning.
* Service Density: Increasing the number of additional services used by a customer significantly reduces churn probability.
* Contract Type: Month-to-month contracts are the strongest predictor of customer defection.

## Installation and Usage
This project uses a Conda environment for dependency management.

1. Create the environment:
   conda create -n ml-churn python=3.10

2. Activate the environment:
   conda activate ml-churn

3. Install dependencies:
   conda install pandas numpy scikit-learn matplotlib seaborn xgboost -c conda-forge
   pip install imbalanced-learn

4. Run the full pipeline:
   python src/initial_profiling.py
   python src/eda.py
   python src/feature_engineering.py
   python src/train_models.py
   python src/optimize_models.py
   python src/final_summary.py
