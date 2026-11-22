# Credit Scoring - Loan Approval Prediction

Task 1 of my **CodeAlpha Machine Learning Internship**.

The idea: look at an applicant's details (income, loan amount, credit history, education, etc.) and predict if their loan gets approved or not.

## What's inside

- `notebook.ipynb` - the whole project, step by step
- `Loan Eligibility Prediction.csv` - dataset ([Kaggle](https://www.kaggle.com/datasets/avineshprabhakaran/loan-eligibility-prediction))
- `requirements.txt`

## Steps I followed

1. Loaded the data and checked missing values / duplicates
2. Did some quick EDA (approval rate by credit history, education, etc.)
3. Dropped `Customer_ID` since it's just an ID
4. Built a preprocessing pipeline: median fill + scaling for numbers, one-hot encoding for text columns
5. Trained Logistic Regression, Decision Tree and Random Forest
6. Compared them using Accuracy, Precision, Recall, F1 and ROC-AUC

## Results

<!-- fill this after running the notebook -->
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| | | | | | |

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Then run all cells top to bottom.

## What I learned

<!-- write 2-3 lines in your own words -->
