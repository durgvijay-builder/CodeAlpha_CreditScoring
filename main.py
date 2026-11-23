from pathlib import Path
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Loan Eligibility Prediction.csv"


def build_pipeline():
    # same preprocessing + model as the notebook, trained on the full dataset
    data = pd.read_csv(CSV_PATH).drop_duplicates().copy()
    data = data.drop(columns=["Customer_ID"], errors="ignore")
    data = data.dropna(subset=["Loan_Status"])

    X = data.drop(columns="Loan_Status")
    y = data["Loan_Status"]

    numeric_columns = X.select_dtypes(include="number").columns
    categorical_columns = X.select_dtypes(exclude="number").columns

    prepare = ColumnTransformer(transformers=[
        ("numbers", Pipeline([
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_columns),
        ("categories", Pipeline([
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_columns),
    ])

    pipe = Pipeline([
        ("prepare", prepare),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


PIPELINE = build_pipeline()


class LoanApplicationInput(BaseModel):
    Gender: str = "Male"
    Married: str = "Yes"
    Dependents: str | int = 1
    Education: str = "Graduate"
    Self_Employed: str = "No"
    Applicant_Income: float = 5468.0
    Coapplicant_Income: float = 1032.0
    Loan_Amount: float = 128.0
    Loan_Amount_Term: float = 360.0
    Credit_History: float = 1.0
    Property_Area: str = "Semiurban"


app = FastAPI(title="Loan Underwriting Workbench")


@app.get("/")
def serve_dashboard():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/style.css")
def serve_style():
    return FileResponse(BASE_DIR / "style.css", media_type="text/css")


@app.get("/tokens.css")
def serve_tokens():
    return FileResponse(BASE_DIR / "tokens.css", media_type="text/css")


@app.get("/base.css")
def serve_base():
    return FileResponse(BASE_DIR / "base.css", media_type="text/css")


@app.post("/predict")
def predict_application(payload: LoanApplicationInput):
    row = payload.model_dump()
    row["Dependents"] = int(str(row["Dependents"]).replace("+", "") or 0)

    df = pd.DataFrame([row])
    pred = PIPELINE.predict(df)[0]
    y_idx = list(PIPELINE.classes_).index("Y")
    prob_yes = float(PIPELINE.predict_proba(df)[0][y_idx])
    return {
        "Loan_Status": pred,
        "probability": round(prob_yes, 4),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
