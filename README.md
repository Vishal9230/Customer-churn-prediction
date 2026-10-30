# Customer Churn Prediction App

## 📌 Project Overview

Customer churn is a critical problem for subscription-based businesses, as retaining existing customers can be more cost-effective than acquiring new ones.

This project builds a Machine Learning system to predict customer churn probability using customer demographics, service usage, subscription details, and billing data.

The project focuses on the complete Machine Learning workflow, including:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- Model training and comparison
- Class imbalance handling
- Hyperparameter tuning using GridSearchCV
- Threshold optimization
- Model explainability using SHAP
- Customer churn risk segmentation
- Model evaluation and export

The goal is to identify customers who are more likely to churn and generate insights that can help businesses develop effective customer-retention strategies.

## 📊 Dataset

This project uses the **Telco Customer Churn Dataset** available on Kaggle.

**Dataset Link:**  
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7,043 customer records initially
- 21 columns, including the target variable
- 7,032 records remaining after data cleaning
- Target variable: `Churn`

### Feature Categories

**Customer Demographics**

- Gender
- Senior Citizen
- Partner
- Dependents

**Account Information**

- Tenure
- Contract Type
- Paperless Billing
- Payment Method

**Services Subscribed**

- Internet Service
- Phone Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

**Billing Information**

- Monthly Charges
- Total Charges

The objective is to predict customer churn probability and support data-driven retention decisions.

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

## ⚙️ Machine Learning Pipeline

### 🔹 Numerical Features

- Tenure
- MonthlyCharges
- TotalCharges

**Processing:**

- Missing-value imputation using the median
- Standard scaling using `StandardScaler`

### 🔹 Categorical Features

- Gender
- Contract Type
- Internet Service
- Payment Method
- Customer demographics
- Service-related features

**Processing:**

- Missing-value imputation using the most frequent value
- One-hot encoding using `OneHotEncoder`

All preprocessing is handled through a Scikit-learn `ColumnTransformer` and integrated into Machine Learning pipelines for consistent data transformation.

## 🤖 Models Used

### 🔹 Logistic Regression

- Baseline classification model
- Provides interpretable probability-based predictions
- Achieves high recall for identifying potential churn customers

### 🔹 Random Forest

- Ensemble learning algorithm
- Captures nonlinear relationships in customer data
- Uses class weighting to address class imbalance
- Hyperparameters optimized using GridSearchCV
- Supports feature importance analysis and SHAP explanations

### 🔹 XGBoost

- Gradient boosting classification algorithm
- Uses `scale_pos_weight` to handle class imbalance
- Captures complex patterns in customer behavior
- Evaluated against Logistic Regression and Random Forest

## 📈 Model Performance

The following results were obtained from the notebook's model comparison.

| Model | ROC-AUC | F1 Score | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.84 | 0.63 | 0.51 | 0.81 |
| Random Forest | 0.85 | 0.63 | 0.55 | 0.75 |
| XGBoost | 0.85 | 0.63 | 0.53 | 0.80 |

### Key Observations

- **Logistic Regression:** Achieves 0.81 recall, identifying a high proportion of customers who churn.
- **Random Forest:** Achieves 0.85 ROC-AUC and 0.55 precision.
- **XGBoost:** Achieves 0.85 ROC-AUC and 0.80 recall.

Random Forest and XGBoost achieve the highest ROC-AUC in the reported comparison, while Logistic Regression achieves the highest recall.

The best model depends on the business objective, particularly the trade-off between identifying churn customers and minimizing false-positive predictions.

## ⚖️ Threshold Optimization

Instead of relying exclusively on the default classification threshold of 0.5, multiple probability thresholds were evaluated.

The project tests thresholds ranging from 0.10 to 0.85 to understand the trade-off between:

- **Precision:** Reducing false-positive predictions.
- **Recall:** Identifying more customers who are likely to churn.
- **F1 Score:** Balancing precision and recall.

For example, at a threshold of 0.45, the Random Forest model achieves approximately:

- Precision: 0.53
- Recall: 0.81
- F1 Score: 0.64

This analysis helps businesses select a threshold based on their customer-retention objectives.

## 🔍 Model Explainability (SHAP)

SHAP (SHapley Additive Explanations) is used to understand how individual features contribute to the Random Forest model's predictions.

The project includes:

- SHAP summary plots
- SHAP feature-importance bar plots
- Feature contribution analysis
- Global model interpretation

These visualizations help identify which customer attributes influence the model's churn predictions.

**Note:** SHAP explains model behavior; it does not establish that a feature directly causes customer churn.

## 📊 Customer Churn Risk Segmentation

The project categorizes customers into three risk levels based on their predicted churn probabilities.

| Churn Probability | Risk Level |
| --- | --- |
| 0%–30% | Low Risk |
| Above 30%–60% | Medium Risk |
| Above 60%–100% | High Risk |

### Risk Distribution

On the notebook's test set, the Random Forest model produces the following distribution:

- **Low Risk:** 956 customers
- **Medium Risk:** 585 customers
- **High Risk:** 569 customers

This segmentation can help businesses prioritize customer-retention activities based on predicted risk.

## 💡 Business Recommendations

Based on the potential applications of churn prediction:

- Identify high-risk customers for targeted retention campaigns.
- Analyze contract types associated with increased churn risk.
- Monitor customers with shorter subscription tenure.
- Investigate billing and service-related concerns.
- Develop personalized retention offers.
- Use predictive insights to prioritize customer engagement.

These recommendations should be validated against actual customer behavior and business outcomes before implementation.

## 💾 Model Export

The trained models are saved using Joblib for potential reuse.

Exported model files:

- `logistic_model.pkl`
- `rf_model.pkl`
- `xgb_model.pkl`

The saved pipelines include preprocessing and model components, allowing consistent transformations when making predictions on new customer data.

## 📌 Conclusion

This project demonstrates an end-to-end Machine Learning workflow for customer churn prediction.

By combining predictive modeling, model evaluation, threshold optimization, explainability, and customer risk segmentation, it demonstrates how Machine Learning can transform customer data into actionable business insights.

The project highlights the importance of evaluating multiple models and selecting an appropriate prediction strategy based on business requirements.

## 👨‍💻 Author

**Vishal Tomar**
