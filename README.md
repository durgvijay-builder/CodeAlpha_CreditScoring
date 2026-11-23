# Credit Scoring - Loan Approval Prediction

Task 1 of my **CodeAlpha Machine Learning Internship**.

This project evaluates whether a loan application gets approved (`Y`) or rejected (`N`) using 11 applicant attributes (credit history, applicant and co-applicant income, requested loan amount, loan term, education, marital status, dependents, employment type, and property area). It includes both a step-by-step Jupyter Notebook and an interactive FastAPI web workbench.

## What's Inside

- `notebook.ipynb` — Complete step-by-step walkthrough: EDA, preprocessing pipeline, model training, evaluation metrics, and test predictions
- `main.py` — FastAPI server that fits the scikit-learn `ColumnTransformer` + `LogisticRegression` pipeline and exposes `/predict` along with the dashboard UI
- `index.html`, `style.css`, `tokens.css`, `base.css` — Interactive Loan Underwriting Workbench with preset profiles, cohort baseline comparisons, and an application ledger
- `Loan Eligibility Prediction.csv` — Dataset of 614 historical loan applications ([Kaggle](https://www.kaggle.com/datasets/avineshprabhakaran/loan-eligibility-prediction))
- `requirements.txt` — Python package dependencies

## Workflow & Engineering Decisions

1. **Data Inspection & Cleaning:** Loaded the 614-row dataset, verified missing values and duplicates, and dropped `Customer_ID` so the model does not memorize arbitrary identifier numbers.
2. **Exploratory Data Analysis (EDA):** Checked approval rates across key cohorts:
   - Credit history met (`1`): **78.9%** approved (412 / 522) vs no documented history (`0`): **10.9%** approved (10 / 92)
   - Property area: **Semiurban 76.8%**, **Urban 65.8%**, **Rural 61.5%**
   - Education: **Graduate 70.8%** vs **Not Graduate 61.2%**
3. **Leak-Free Preprocessing Pipeline:** Built a `ColumnTransformer` that applies median imputation + `StandardScaler` to numeric columns and most-frequent imputation + `OneHotEncoder` to categorical columns inside each model pipeline.
4. **Stratified 80/20 Benchmark:** Trained and evaluated Logistic Regression, Random Forest (200 trees), and Decision Tree (`max_depth=5`) on a stratified 80/20 train/test split (491 train, 123 test).

## Results (80/20 Stratified Test Split)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.8049** | 0.8020 | **0.9529** | **0.8710** | **0.8012** |
| **Random Forest (200 trees)** | 0.7805 | 0.7959 | 0.9176 | 0.8525 | 0.7861 |
| **Decision Tree (max_depth=5)** | **0.8049** | **0.8144** | 0.9294 | 0.8681 | 0.6735 |

## How to Run

```bash
pip install -r requirements.txt

# 1. Run the Jupyter Notebook
jupyter notebook notebook.ipynb

# 2. Or start the interactive FastAPI Underwriting Workbench
python main.py
```

Then open `http://127.0.0.1:8000` in your browser.

## What I Learned

- **Accuracy alone is misleading on imbalanced targets:** Because 68.7% of applications in the dataset are approved (`422 Y` vs `192 N`), looking at Precision, Recall, F1, and **ROC-AUC** together gives a much clearer picture of how well each classifier separates risky vs creditworthy applicants.
- **Pipelines prevent data leakage:** Wrapping `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` inside a `ColumnTransformer` pipeline ensures test rows are transformed using only training-set statistics.
