🏦 Credit Risk Modeling
This repository contains credit risk modeling projects that aim to assess the probability of loan default. It includes two different approaches:
Home Credit Default Prediction (machine learning pipeline using LightGBM)
Probability of Default (PD) Model (logistic regression with validation pipeline)
Together, they demonstrate both a modern ML-driven approach and a traditional, interpretable statistical approach to credit risk.

## 🔎 Project Overview  

### 1. Home Credit Risk Model (ML Pipeline)  
- **Data**: Customer-level features including personal income, housing details, and credit history  
- **Model**: LightGBM binary classifier  
- **Feature Engineering**: Aggregation of historical records, categorical encoding, handling of missing values  

**Business Metrics**:  
- **Default Rate Reduction**: Identify risky applicants to reduce losses from defaults  
- **Approval Efficiency**: Automate credit risk scoring for faster decisions  
- **Profit Impact**: Improve loan portfolio health and profitability  

**Insights**:  
- **Income Type**: Self-employed applicants show higher default risk  
- **Credit History**: Longer overdue payments strongly correlate with defaults  
- **Housing Stability**: Home ownership is associated with lower risk  

---

### 2. Credit Scoring Model (LendingClub Loan Data)  
- **Dataset**: LendingClub loan data, including borrower demographics, financial attributes, loan details, and repayment outcomes.
  https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv
- **Preprocessing**:  
  - Handle missing values  
  - Encode categorical variables into dummy variables  
  - Clean and prepare the dataset for modeling  

- **Model**: Logistic Regression (industry-standard approach for credit scoring / PD estimation)  
  - Outputs probability of default (PD) for each loan application  
  - Provides interpretable coefficients to understand the impact of borrower features  

- **Validation**:  
  - Confusion Matrix, Accuracy, Precision, Recall, F1-score  
  - ROC Curve & AUC to evaluate discriminatory power  
  - Cross-validation to ensure model robustness and generalization  
