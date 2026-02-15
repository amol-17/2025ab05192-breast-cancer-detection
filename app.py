from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
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
    roc_curve,
)

st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

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


def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILE_MAP[name])


def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.title("Breast Cancer Classification")
    st.write(
        "This app evaluates six ML models on the UCI Breast Cancer Wisconsin (Diagnostic) dataset. "
        "You can upload your own test CSV or use the bundled test split."
    )

    if not MODEL_DIR.exists():
        st.error("Model directory not found. Please run train_and_save_models.py first.")
        st.stop()

    metadata_path = MODEL_DIR / "dataset_metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        st.info(
            f"Dataset: {meta['dataset_name']} | "
            f"{meta['instances']} samples, {meta['features']} features | "
            f"Classes: {', '.join(meta['target_names'])}"
        )

    uploaded_file = st.file_uploader("Upload test CSV (optional)", type=["csv"])
    default_test_df = pd.read_csv(DATA_DIR / "test_data.csv")

    if uploaded_file is not None:
        eval_df = pd.read_csv(uploaded_file)
        st.success("Uploaded file loaded.")
    else:
        eval_df = default_test_df.copy()
        st.caption("Using the bundled test split.")

    feature_cols = [c for c in default_test_df.columns if c != "target"]

    missing = [c for c in feature_cols if c not in eval_df.columns]
    if missing:
        st.error("Uploaded CSV is missing columns: " + ", ".join(missing))
        st.stop()

    has_labels = "target" in eval_df.columns
    X_eval = eval_df[feature_cols]
    y_true = eval_df["target"] if has_labels else None

    st.divider()
    st.subheader("All-Model Comparison")

    all_metrics = []
    model_predictions = {}

    for name in MODEL_FILE_MAP:
        model = load_model(name)
        preds = model.predict(X_eval)
        probs = model.predict_proba(X_eval)[:, 1]
        model_predictions[name] = {"preds": preds, "probs": probs}

        if has_labels:
            metrics = evaluate(y_true, preds, probs)
            metrics["Model"] = name
            all_metrics.append(metrics)

    if has_labels and all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df = metrics_df[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
        metrics_df = metrics_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

        styled = metrics_df.style.format(
            {col: "{:.4f}" for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}
        ).background_gradient(cmap="Greens", subset=["Accuracy", "AUC", "F1"])
        st.dataframe(styled, use_container_width=True)

        st.subheader("Confusion Matrices")
        cols = st.columns(3)
        for i, name in enumerate(MODEL_FILE_MAP):
            with cols[i % 3]:
                cm = confusion_matrix(y_true, model_predictions[name]["preds"])
                fig, ax = plt.subplots(figsize=(3.5, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_title(name, fontsize=10)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                plt.close(fig)

        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(7, 5))
        for name in MODEL_FILE_MAP:
            fpr, tpr, _ = roc_curve(y_true, model_predictions[name]["probs"])
            auc_val = roc_auc_score(y_true, model_predictions[name]["probs"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Comparison")
        ax.legend(fontsize=8, loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Per-Model Deep Dive")

    selected = st.selectbox("Pick a model to inspect", list(MODEL_FILE_MAP.keys()))
    preds = model_predictions[selected]["preds"]
    probs = model_predictions[selected]["probs"]

    if has_labels:
        metrics = evaluate(y_true, preds, probs)

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        c1.metric("AUC", f"{metrics['AUC']:.4f}")
        c2.metric("Precision", f"{metrics['Precision']:.4f}")
        c2.metric("Recall", f"{metrics['Recall']:.4f}")
        c3.metric("F1", f"{metrics['F1']:.4f}")
        c3.metric("MCC", f"{metrics['MCC']:.4f}")

        st.write("**Classification Report**")
        report = classification_report(y_true, preds, output_dict=True)
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)
    else:
        st.warning("No target column found. Upload a CSV with a 'target' column to see metrics.")

    st.subheader("Prediction Preview")
    preview = eval_df.copy()
    preview["prediction"] = preds
    st.dataframe(preview.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
