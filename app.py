from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


st.set_page_config(page_title="ML Assignment 2 - Breast Cancer Classifier", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_FILE_MAP = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
    "XGBoost (Ensemble)": "xgboost_ensemble.pkl",
}


def load_model(selected_model: str):
    model_path = MODEL_DIR / MODEL_FILE_MAP[selected_model]
    return joblib.load(model_path)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


def main() -> None:
    st.title("Machine Learning Assignment 2 - Classification Models")
    st.write(
        "Dataset: UCI Breast Cancer Wisconsin (Diagnostic). "
        "Upload test CSV data or use bundled test split."
    )

    if not MODEL_DIR.exists():
        st.error("Model directory is missing. Run model/train_and_save_models.py first.")
        st.stop()

    metadata_path = MODEL_DIR / "dataset_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        st.info(
            f"Dataset instances: {metadata['instances']} | "
            f"Features: {metadata['features']} | "
            f"Target labels: {', '.join(metadata['target_names'])}"
        )

    st.subheader("Model Comparison (Precomputed on held-out test set)")
    metrics_table_path = MODEL_DIR / "model_metrics.csv"
    if metrics_table_path.exists():
        metrics_df = pd.read_csv(metrics_table_path)
        st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Interactive Evaluation")

    selected_model = st.selectbox("Select model", list(MODEL_FILE_MAP.keys()))
    uploaded_file = st.file_uploader("Upload CSV test data", type=["csv"])

    default_test_df = pd.read_csv(DATA_DIR / "test_data.csv")

    if uploaded_file is not None:
        eval_df = pd.read_csv(uploaded_file)
        st.success("Uploaded dataset loaded successfully.")
    else:
        eval_df = default_test_df.copy()
        st.caption("Using bundled test split (data/test_data.csv).")

    expected_feature_cols = [col for col in default_test_df.columns if col != "target"]

    missing_cols = [col for col in expected_feature_cols if col not in eval_df.columns]
    if missing_cols:
        st.error(
            "Uploaded CSV is missing required feature columns: " + ", ".join(missing_cols)
        )
        st.stop()

    model = load_model(selected_model)
    X_eval = eval_df[expected_feature_cols]
    y_pred = model.predict(X_eval)

    st.write(f"Predictions generated using **{selected_model}**")

    if "target" in eval_df.columns:
        y_true = eval_df["target"]
        y_prob = model.predict_proba(X_eval)[:, 1]
        metric_values = compute_metrics(y_true, y_pred, y_prob)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metric_values['Accuracy']:.4f}")
            st.metric("AUC", f"{metric_values['AUC']:.4f}")
        with col2:
            st.metric("Precision", f"{metric_values['Precision']:.4f}")
            st.metric("Recall", f"{metric_values['Recall']:.4f}")
        with col3:
            st.metric("F1", f"{metric_values['F1']:.4f}")
            st.metric("MCC", f"{metric_values['MCC']:.4f}")

        st.subheader("Confusion Matrix")
        plot_confusion_matrix(y_true, y_pred)

        st.subheader("Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    else:
        st.warning(
            "No 'target' column in uploaded file. Metrics and confusion matrix require labels."
        )

    preview_df = eval_df.copy()
    preview_df["prediction"] = y_pred

    st.subheader("Prediction Preview")
    st.dataframe(preview_df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
