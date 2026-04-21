# Annual-Income-Estimator-KNN-RandomForest
Estimating a bank customer's annual salary from their transaction behavior, demographics, and financial profile using machine learning. The project covers the full pipeline from raw transaction data to income predictions on new customers.

## Business Problem
Banks often need to estimate a customer's income without requiring them to self-report it. This model infers annual salary from observable transaction patterns — spending behavior, balance levels, payment timing, and demographic signals — enabling more accurate credit decisions, product targeting, and risk profiling.

## Dataset
Training data: income_est.xlsx — historical transaction records with known annual salaries
Deployment data: income_deploy.xlsx — new customers to be scored
Target variable: annual_salary

Key features include transaction amounts, account balance, merchant data, customer demographics (age, gender), payment behavior, and timing signals.

## Pipeline Overview

- Data Loading & Exploration — inspect shape, nulls, distributions
- Data Cleaning — drop ID/admin columns, impute nulls with mean/mode
- Feature Engineering — create transaction-aware features:
    - balance_to_amount_ratio — normalized spending relative to balance
    - is_salary_transaction — flags PAY/SALARY transactions
    - age_balance_interaction — wealth-age combined signal
    - amount_per_payment_period — normalized transaction size
    - is_weekend_spender, is_business_hours — behavioral timing flags
    - is_card_present, is_debit, is_authorized_only — transaction type flags

- Model-Specific Preprocessing — separate pipelines for RF and KNN
- Outlier Treatment — IQR capping for KNN pipeline
- Normality Testing — KS test per feature
- Feature Selection — Spearman correlation, inter-correlation check, VIF analysis
- Encoding & Scaling — ordinal/label encoding + StandardScaler for KNN
- Modeling — baseline KNN and Random Forest regressors
- Hyperparameter Optimization — Optuna (15 trials for KNN, 30 for RF)
- Feature Importance Filtering — retain features with importance > 0.01
- Model Comparison — R2 scores across all models
- Deployment — predict annual salary for new customers


## Evaluation Metrics

- R2 (primary) — proportion of variance in salary explained by the model
- MAE — average absolute prediction error in salary units
- RMSE — root mean squared error, penalizes large errors more


