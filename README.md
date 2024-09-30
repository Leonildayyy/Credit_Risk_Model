Home Credit Credit Risk Model

This project implements a machine learning pipeline to predict credit risk using Home Credit loan data. The goal is to classify loan applications based on the likelihood of default.

Overview

Data: Includes personal income, housing details, and credit history.
Model: LightGBM is used to train a binary classifier.
Feature Engineering: Aggregation of historical records, typecasting, and handling of categorical variables.

Business Metrics

Default Rate Reduction: By accurately predicting risky applicants, the model helps minimize the default rate, which can lead to significant cost savings.
Approval Efficiency: Faster, more accurate loan approvals by automating the risk assessment process.
Profit Impact: Improved loan portfolio health and increased profitability by lending to low-risk applicants.

Insights

Income Type as a Key Predictor: Self-employed individuals tend to have a higher risk, which can be factored into more stringent lending criteria.
Credit History Significance: Longer overdue payments (e.g., pmts_dpdvalue_108P_over31) are strongly correlated with higher default risk.
Housing Stability: Applicants with more stable housing conditions (e.g., home ownership) are less likely to default.
