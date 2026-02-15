from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class ModelResult:
    model_name: str
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    mcc: float


def build_preprocessor(feature_names: list[str], use_scaler: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_names)],
        remainder="drop",
    )


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series, model_name: str) -> ModelResult:
    return ModelResult(
        model_name=model_name,
        accuracy=accuracy_score(y_true, y_pred),
        auc=roc_auc_score(y_true, y_prob),
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        mcc=matthews_corrcoef(y_true, y_pred),
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    feature_names = X_train.columns.tolist()

    model_specs: list[tuple[str, Pipeline]] = [
        (
            "Logistic Regression",
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(feature_names, use_scaler=True)),
                    (
                        "classifier",
                        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                    ),
                ]
            ),
        ),
        (
            "Decision Tree",
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(feature_names, use_scaler=False)),
                    ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
                ]
            ),
        ),
        (
            "kNN",
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(feature_names, use_scaler=True)),
                    ("classifier", KNeighborsClassifier(n_neighbors=7)),
                ]
            ),
        ),
        (
            "Naive Bayes",
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(feature_names, use_scaler=False)),
                    ("classifier", GaussianNB()),
                ]
            ),
        ),
        (
            "Random Forest (Ensemble)",
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(feature_names, use_scaler=False)),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=400,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "XGBoost (Ensemble)",
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(feature_names, use_scaler=False)),
                    (
                        "classifier",
                        XGBClassifier(
                            n_estimators=350,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            random_state=RANDOM_STATE,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]

    results: list[dict[str, float | str]] = []
    confusion_matrices: dict[str, list[list[int]]] = {}

    for model_name, model_pipeline in model_specs:
        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob, model_name)
        results.append(asdict(metrics))

        confusion_matrices[model_name] = confusion_matrix(y_test, y_pred).tolist()

        model_file_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_file_path = MODEL_DIR / f"{model_file_name}.pkl"
        joblib.dump(model_pipeline, model_file_path)

    metrics_df = pd.DataFrame(results)
    metrics_df.sort_values(by="accuracy", ascending=False, inplace=True)
    metrics_df.to_csv(MODEL_DIR / "model_metrics.csv", index=False)

    split_test_df = X_test.copy()
    split_test_df["target"] = y_test.values
    split_test_df.to_csv(DATA_DIR / "test_data.csv", index=False)

    split_train_df = X_train.copy()
    split_train_df["target"] = y_train.values
    split_train_df.to_csv(DATA_DIR / "train_data.csv", index=False)

    metadata = {
        "dataset_name": "UCI Breast Cancer Wisconsin (Diagnostic)",
        "instances": int(X.shape[0]),
        "features": int(X.shape[1]),
        "target_names": dataset.target_names.tolist(),
        "feature_names": feature_names,
        "random_state": RANDOM_STATE,
    }

    with open(MODEL_DIR / "dataset_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with open(MODEL_DIR / "confusion_matrices.json", "w", encoding="utf-8") as file:
        json.dump(confusion_matrices, file, indent=2)

    print("Training complete. Files saved in:")
    print(f"- {MODEL_DIR}")
    print(f"- {DATA_DIR}")


if __name__ == "__main__":
    main()
