Customer Churn Prediction

Predicts which telecom customers are likely to leave (churn), compares three machine learning models, and explains each prediction with SHAP. Includes an interactive Streamlit app.

Live app: http://localhost:8501/

Problem

Winning a new customer costs far more than keeping an existing one. If a telecom company can spot customers who are about to leave, it can reach out with an offer before they go. This project builds a model that gives each customer a churn probability and shows which factors pushed it up or down.

Dataset

Telco Customer Churn (Kaggle)

7,043 customers, 21 columns
Features cover demographics, services used (internet, phone, streaming, tech support), contract type, billing and payment method, tenure, and monthly and total charges
Target: Churn (Yes / No)
Imbalanced: about 27% of customers churned
Approach
Cleaning: TotalCharges is stored as text; converted to numeric and dropped the 11 rows where it was blank.
Split: 70/30 train-test split using stratify=y so both sets keep the same churn rate.
Preprocessing pipeline: median imputation + standard scaling for numeric columns, most-frequent imputation + one-hot encoding for categorical columns, combined with ColumnTransformer. Keeping preprocessing inside the pipeline means the exact same steps run in training and in the app.
Class imbalance: class_weight='balanced' for Logistic Regression and Random Forest, scale_pos_weight for XGBoost.
Models: Logistic Regression, Random Forest (tuned with GridSearchCV, 3-fold, scored on ROC AUC), and XGBoost.
Evaluation: ROC AUC, F1, precision and recall on the held-out test set, plus a precision-recall trade-off across decision thresholds.
Explainability: SHAP values show which features drive churn overall and for a single customer.
Results

Test set of 2,110 customers:

Model	ROC AUC	F1	Precision	Recall
Logistic Regression	0.84	0.63	0.51	0.81
Random Forest	0.85	0.63	0.55	0.75
XGBoost	0.85	0.63	0.53	0.80

All three models perform similarly (ROC AUC around 0.85). Recall was prioritised because missing a customer who is about to leave costs more than contacting one who would have stayed. Logistic Regression and XGBoost catch about 80% of churners. Lowering the decision threshold raises recall further at the cost of precision (see the threshold analysis in the notebook).

Streamlit App
Choose a model (Logistic Regression, Random Forest or XGBoost)
Enter a customer's details and get a churn prediction with probability
See a SHAP explanation of what pushed that prediction up or down
A Model Insights tab compares the three models
Project Structure
customer-churn-prediction/
├── app/
│   └── app.py              # Streamlit app
├── data/
│   └── Telco_Customer_Churn.csv
├── models/
│   ├── logistic_model.pkl
│   ├── rf_model.pkl
│   └── xgb_model.pkl
├── notebooks/
│   └── CustomerChurnPrediction.ipynb
├── requirements.txt
└── README.md
Run Locally
bash
git clone https://github.com/Vishal9230/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app/app.py

Run the last command from the main project folder so the app can find the models/ folder.

Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, Streamlit

Limitations and Next Steps
The train-test split has no fixed random seed, so metrics shift slightly between runs. The numbers above come from the run that produced the saved models.
The dataset comes from a single company and a snapshot in time, so the model would need retraining on fresh data before real use.
Possible next steps: fix the random seed, add cross-validated comparison of all three models, tune the decision threshold to a business cost, and add feature engineering such as tenure groups.
