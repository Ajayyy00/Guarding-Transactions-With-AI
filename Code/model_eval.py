import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_behavior.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Loading dataset...")
data = pd.read_csv('creditcard.csv')

logger.info("Cleaning data...")
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
logger.info(f"Dataset shape after cleaning: {data.shape}")

logger.info("Performing feature engineering...")
data.loc[:, 'Log_Amount'] = np.log1p(data['Amount'])
data.loc[:, 'Hour'] = (data['Time'] // 3600) % 24
data.loc[:, 'Time_Diff'] = data['Time'].diff().fillna(0)
features = ['V' + str(i) for i in range(1, 29)] + ['Log_Amount', 'Hour', 'Time_Diff']
X = data[features]
y = data['Class']
logger.info(f"Features used: {features}")

logger.info("Loading scaler and model...")
scaler = joblib.load('scaler.pkl')
model = joblib.load('best_fraud_detection_model.pkl')

logger.info("Standardizing features...")
X_scaled = scaler.transform(X)
logger.info(f"Feature matrix shape after scaling: {X_scaled.shape}")

logger.info("Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
logger.info(f"Test set shape: {X_test.shape}")

logger.info("Evaluating model on test set...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

logger.info("Evaluation Metrics:")
logger.info(f"Precision: {precision:.4f}")
logger.info(f"Recall: {recall:.4f}")
logger.info(f"F1-Score: {f1:.4f}")
logger.info(f"ROC-AUC: {roc_auc:.4f}")
logger.info(f"Confusion Matrix:\n{conf_matrix}")

logger.info("Generating visualizations...")

plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=data, palette='Blues')
plt.title('Class Distribution (0: Legitimate, 1: Fraudulent)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.yscale('log')  
plt.savefig('class_distribution.png')
plt.close()
logger.info("Saved Class Distribution as 'class_distribution.png'")

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve.png')
plt.close()
logger.info("Saved ROC curve as 'roc_curve.png'")

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label='Precision-Recall Curve', color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig('precision_recall_curve.png')
plt.close()
logger.info("Saved Precision-Recall curve as 'precision_recall_curve.png'")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Legitimate', 'Fraudulent'], yticklabels=['Legitimate', 'Fraudulent'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()
logger.info("Saved Confusion Matrix as 'confusion_matrix.png'")

if hasattr(model, 'feature_importances_'):
    logger.info("Generating feature importance plot...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(features)), importances[indices], align='center', color='blue')
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    logger.info("Saved Feature Importance as 'feature_importance.png'")
else:
    logger.info("Model does not support feature importance.")
