# Telco Customer Churn Prediction and Model Comparison

## Project Overview
This project focuses on predicting customer churn for a telecommunications service provider. In subscription-based industries, retaining existing customers is significantly more cost-effective than acquiring new ones. This system implements machine learning models to identify high-risk customers, enabling proactive retention strategies.

## Performance Targets
The primary goal of this project was to prioritize Recall over overall Accuracy.
* Target Recall: Higher than 75%
* Achieved Recall: 82.62% (Logistic Regression - Weighted)
* Achieved Accuracy: 72.46%
* AUC-ROC: 0.85

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

## EXPLAIN OF THE RESULTS

  To explain these scores, think of them as answers to different questions about how "smart" the model is.

  In our project, a "Positive" result means the model predicted the customer WILL CHURN (leave).

  1. Recall (Our Most Important Score: 82.62%)
   * The Question: "Out of everyone who actually left, how many did we successfully catch?"
   * Why it matters: In churn prediction, missing a customer who is about to leave is expensive (you lose their future revenue).
   * In Simple Terms: If 100 people left, our model successfully "sounded the alarm" for 83 of them.
 
  2. Accuracy (72.46%)
   * The Question: "Overall, how many times was the model right (both about people staying and people leaving)?"
   * Why it matters: It gives a general sense of performance.
   * In Simple Terms: Out of 100 total customers, the model correctly guessed the outcome for about 72 of them.
 
  3. Precision (48.89% - from our optimized model)
   * The Question: "When the model says someone is going to leave, how often is it actually right?"
   * Why it matters: If precision is too low, you waste money sending "Please stay" discounts to people who were never going to leave anyway (False Alarms).
   * In Simple Terms: When our model points at 10 people and says "They will leave," about 5 of them actually would have.
 
  4. F1-Score (61.43%)
   * The Question: "What is the balance between Precision and Recall?"
   * Why it matters: It's hard to have both 100% Precision and 100% Recall. If you try to catch everyone (High Recall), you get more false alarms (Low Precision).
   * In Simple Terms: This is the "Combined Score" that proves the model is stable and not just guessing.
 
  5. AUC (0.85)
   * The Question: "How good is the model at ranking customers from 'Lowest Risk' to 'Highest Risk'?"
   * Why it matters: A score of 0.5 is like flipping a coin (useless). 1.0 is a perfect psychic.
   * In Simple Terms: At 0.85, our model is "Excellent." It is very good at putting the real leavers at the top of the list so the marketing team can call them first.

---

  When the professor asks, "Why is your accuracy only 73% but your recall is 78%?"
  > Answer: "We intentionally optimized for Recall. We would rather have a few 'False Alarms' (Precision) than miss a customer who is actually leaving (Recall), because losing a customer is much more expensive
  than sending an unnecessary discount."
