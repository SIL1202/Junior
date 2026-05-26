# Telco Customer Churn Prediction & Model Comparison (Final Report)
> **Status:** Completed (Weeks 14-16)
> **Primary Objective:** achieve Recall > 75% to identify at-risk customers.

---

## 1. Group Members
* Student A (ID: XXXXXXX) - *Data Preprocessing & EDA*
* Student B (ID: XXXXXXX) - *Model Training & Optimization*
* Student C (ID: XXXXXXX) - *Feature Engineering & Analysis*
* Student D (ID: XXXXXXX) - *Presentation & Reporting*

---

## 2. Project Overview
* **Goal:** Build a proactive retention system using Machine Learning to predict customer churn.
* **Key Metric:** **Recall** (Sensitivity). Failing to detect a leaver (False Negative) is the highest business cost.
* **Dataset:** 7,043 customer profiles with 21 attributes (Demographics, Services, Financials).

---

## 3. Methodology & Implementation
We implemented and compared three models with advanced optimization techniques:
1.  **Baseline Models:** Logistic Regression, Random Forest, XGBoost (initial training).
2.  **Optimization Techniques:** 
    *   **Class Weighting:** Adjusted the loss function to penalize misclassifying the minority (Churn) class.
    *   **SMOTE:** Synthetic Minority Over-sampling Technique to balance the dataset.
    *   **Feature Engineering:** One-Hot Encoding, Standard Scaling, and **Service Density** calculation.

---

## 4. Final Results (Model Comparison)
The **Logistic Regression (Weighted)** model was selected as the final solution.

| Model | Accuracy | **Recall (Target >75%)** | AUC |
| :--- | :--- | :--- | :--- |
| **Logistic Regression (Weighted)** | **73.81%** | **78.34%** ✅ | **0.84** |
| Logistic Regression (SMOTE) | 73.53% | 70.32% | 0.82 |
| XGBoost (Weighted) | 75.73% | 69.52% | 0.82 |
| Random Forest (Baseline) | 79.13% | 50.00% | 0.82 |

---

## 5. Key Business Insights
1.  **The "Sticky" Effect (Service Density):** Churn rate drops from **43.7%** (0 services) to **5.2%** (8 services). Multi-service integration is the strongest retention tool.
2.  **The "Danger Zone" (Tenure):** Median tenure for churners is only **10 months**. Retention efforts must focus on the first year of the customer lifecycle.
3.  **Contract Influence:** "Month-to-month" contracts are the primary driver of churn. Incentivizing 1-year or 2-year contracts can drastically reduce defection.

---

## 6. Project Structure
*   `src/`: Python scripts for data profiling, EDA, engineering, and training.
*   `reports/figures/`: Visual charts (ROC curves, Feature Importance, EDA).
*   `models/`: Saved `.joblib` files for the final deployment.
*   `data/processed/`: Cleaned and scaled training/testing sets.

---
*Final Documentation generated for Gemini-CLI presentation support. Project Complete.*
