# Customer Churn Prediction: Telecommunications Case Study

**Author:** Abin J Antony

## 📌 Project Overview
This repository contains an end-to-end Machine Learning pipeline developed to predict customer churn for a telecommunications company. By identifying customers with a high probability of leaving, the business can proactively target them with retention offers, thereby reducing revenue leakage.

## 📊 Dataset Description
The dataset encompasses customer behavior and demographic information:
*   **Demographics:** Gender, Senior Citizen status, Partner, and Dependents.
*   **Account Information:** Tenure, Contract type, Paperless Billing, Payment Method, Monthly Charges, and Total Charges.
*   **Services Subscribed:** Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, and Streaming Movies.
*   **Target Variable:** `Churn` (Yes/No)

## 🛠️ Methodology
The project follows a structured Data Science workflow:
1.  **Exploratory Data Analysis (EDA):** Analyzed feature distributions, correlations, and relationships with the target variable to uncover churn patterns.
2.  **Data Preprocessing:** Imputed missing values, encoded categorical variables into numeric representations, and scaled numerical features using `StandardScaler`.
3.  **Feature Engineering:** Created high-impact features to improve predictive power:
    *   `Tenure_Contract`: Captures the relationship between loyalty and contract length.
    *   `CustomerLifetimeValue`: Estimates the total value brought by the customer (`Tenure` * `MonthlyCharges`).
4.  **Model Development:** Trained and evaluated baseline classifiers including Logistic Regression, Random Forest, and Gradient Boosting.
5.  **Hyperparameter Tuning:** Applied `GridSearchCV` with 5-fold cross-validation to optimize the Gradient Boosting model (tuning `learning_rate`, `max_depth`, and `n_estimators`).

## 🚀 Results
The **Tuned Gradient Boosting Classifier** emerged as the best-performing model:
*   **ROC-AUC Score:** 0.8522
*   **Accuracy:** 81%
*   **Precision (Churners):** 0.69
*   The model successfully balances precision and recall, ensuring retention efforts are accurately targeted at high-risk customers without excessive false positives.

## 💡 Business Recommendations
Based on the feature importance analysis, the primary drivers of churn are Contract Type, Tenure, and Monthly Charges.
1.  **Incentivize Long-Term Contracts:** Offer premium feature upgrades or discounts to transition month-to-month customers into 1-year or 2-year contracts.
2.  **Targeted Onboarding:** Implement proactive engagement during the first few months of a customer's lifecycle, as low-tenure customers exhibit the highest churn risk.
3.  **Evaluate Pricing Strategies:** Consider bundled services or targeted discounts for customers in high-monthly-charge brackets to mitigate price sensitivity.

## 💻 Tech Stack
*   **Language:** Python
*   **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
