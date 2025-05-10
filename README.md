# Guarding-Transactions-With-AI
DataSet link :https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download


# Fraud Detection Model Evaluation Report
Generated on: 2025-05-10 05:52:09

## Dataset
- Source: Kaggle Credit Card Fraud Detection Dataset
- Test Set Size: 269540 transactions
- Features: V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Log_Amount, Hour, Time_Diff

## Model
- Type: RandomForestClassifier
- Trained Features: 31 (V1-V28, Log_Amount, Hour, Time_Diff)

## Evaluation Metrics
- **Accuracy**: 1.0000
- **Precision**: 0.9825
- **Recall**: 1.0000
- **F1-Score**: 0.9912
- **ROC-AUC**: 1.0000

## Confusion Matrix
```
[[269083      8]
 [     0    449]]
```
- True Negatives (Legitimate, Legitimate): 269083
- False Positives (Legitimate, Fraudulent): 8
- False Negatives (Fraudulent, Legitimate): 0
- True Positives (Fraudulent, Fraudulent): 449

## Visualizations
- ROC Curve: `roc_curve.png`
- Precision-Recall Curve: `precision_recall_curve.png`
- Confusion Matrix: `confusion_matrix.png`

## Interpretation
- **Recall**: High recall indicates the model detects most fraudulent transactions, critical for fraud prevention.
- **F1-Score**: Balances precision and recall, accounting for false positives.
- **ROC-AUC**: Measures the model's ability to distinguish between classes.
- **Confusion Matrix**: Shows the distribution of correct and incorrect predictions.

For deployment, ensure inputs match the 31 features used in training.
