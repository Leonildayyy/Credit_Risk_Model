# Credit Risk Modeling in Python  

This project is based on the Udemy course **"A Complete Data Science Case Study: Preprocessing, Modeling, Model Validation and Maintenance in Python"**.  

The goal of the project is to build a **Probability of Default (PD) model** for credit risk, following the full data science pipeline.  

## 🔎 Workflow  

### Data Preprocessing (`Data_Processing.ipynb`)  
- Load raw credit risk dataset  
- Handle missing values  
- Encode categorical variables (dummy/one-hot encoding)  
- Clean and prepare the dataset for modeling  

### Modeling (`PD_Model.ipynb`)  
- Logistic Regression as the Probability of Default (PD) model  
- Train/test split for evaluation  
- Model training and probability prediction  

### Validation (`PD_Model.ipynb`)  
- Confusion Matrix, Accuracy, Precision, Recall, F1-score  
- ROC Curve & AUC  
- Cross-validation for model robustness  

---

## 📊 Key Points  
- The **final PD model** is built with **Logistic Regression**.  
- The focus is on a clear **end-to-end pipeline**: from preprocessing → modeling → validation.  
- **No feature scaling/normalization** is applied, as the model is primarily built on encoded categorical and numerical inputs directly.  
