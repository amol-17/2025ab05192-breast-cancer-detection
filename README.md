# Machine Learning Assignment 2 - Classification and Streamlit Deployment

## a) Problem statement
Build a complete end-to-end machine learning classification workflow on one public dataset by:
- implementing six required ML classification models on the same dataset,
- evaluating each model using Accuracy, AUC, Precision, Recall, F1, and MCC,
- creating an interactive Streamlit app for model evaluation,
- preparing deployment-ready project artifacts.

## b) Dataset description
- **Dataset name:** UCI Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository (also available through `sklearn.datasets.load_breast_cancer`)
- **Task type:** Binary classification (malignant vs benign)
- **Total instances:** 569
- **Total input features:** 30 (numeric)
- **Target classes:** 0 = malignant, 1 = benign
- **Assignment constraints check:** Meets minimum 12 features and 500 instances

## c) Models used
The following 6 models are implemented on the same train/test split:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison Table (Evaluation Metrics)
The exact values below are generated from `model/model_metrics.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9561 | 0.9934 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| XGBoost (Ensemble) | 0.9561 | 0.9950 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

### Observations about model performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall in this run, with highest accuracy/F1 and very strong MCC, indicating balanced class prediction. |
| Decision Tree | Lowest scores among all models, likely due to overfitting and weaker generalization on unseen samples. |
| kNN | Second-best by accuracy with perfect recall, showing strong sensitivity for positive class detection. |
| Naive Bayes | Reliable baseline with good AUC, but lower overall accuracy than logistic regression/kNN/ensembles. |
| Random Forest (Ensemble) | Strong balanced performance and high AUC, improving clearly over a single decision tree. |
| XGBoost (Ensemble) | Comparable to random forest in accuracy with very high AUC and recall, making it a robust ensemble option. |

## Project structure
```text
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- data/
│   │-- train_data.csv
│   │-- test_data.csv
│-- model/
│   │-- train_and_save_models.py
│   │-- *.pkl
│   │-- model_metrics.csv
│   │-- dataset_metadata.json
│   │-- confusion_matrices.json
```

## How to run
1. Train and save artifacts:
   ```bash
   python model/train_and_save_models.py
   ```
2. Launch Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Streamlit features implemented
- CSV dataset upload option (recommended: test data)
- Model selection dropdown (all 6 models)
- Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report
- Prediction preview table

## Notes for submission
- Include GitHub repository link and Streamlit app link in final submission PDF.
- Include screenshot showing execution on BITS Virtual Lab.
- Include this README content in the final submission PDF.
