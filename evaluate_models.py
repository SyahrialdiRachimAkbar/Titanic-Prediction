#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import os

# Load preprocessed data
processed_data_path = "/home/ubuntu/data_analysis_project/processed_data.csv"
df_processed = pd.read_csv(processed_data_path)

# Separate features (X) and target (y)
X = df_processed.drop("Survived", axis=1)
y = df_processed["Survived"]

# Split data into training and testing sets (using the same split as training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load trained models
model_dir = "/home/ubuntu/data_analysis_project/models"
model_files = {
    "Logistic Regression": os.path.join(model_dir, "logistic_regression.joblib"),
    "Random Forest": os.path.join(model_dir, "random_forest.joblib"),
    "Gradient Boosting": os.path.join(model_dir, "gradient_boosting.joblib")
}

models = {name: joblib.load(path) for name, path in model_files.items()}

# --- Model Evaluation --- 

evaluation_results = {}
evaluation_dir = "/home/ubuntu/data_analysis_project/evaluation"
os.makedirs(evaluation_dir, exist_ok=True)

plt.figure(figsize=(10, 8))

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probabilities for ROC curve

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    evaluation_results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC AUC": roc_auc
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    cm_path = os.path.join(evaluation_dir, f'{name.replace(" ", "_").lower()}_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"  Confusion Matrix saved to {cm_path}")

    # Plot ROC Curve (add to combined plot)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

# Finalize combined ROC Curve plot
plt.plot([0, 1], [0, 1], "k--", label="Random Chance") # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curves")
plt.legend(loc="lower right")
roc_path = os.path.join(evaluation_dir, "combined_roc_curve.png")
plt.savefig(roc_path)
plt.close()
print(f"\nCombined ROC Curve saved to {roc_path}")

# Save evaluation metrics to a file
evaluation_df = pd.DataFrame(evaluation_results).T # Transpose for better readability
evaluation_file_path = os.path.join(evaluation_dir, "evaluation_metrics.csv")
evaluation_df.to_csv(evaluation_file_path)
print(f"\nEvaluation metrics saved to {evaluation_file_path}")

print("\nModel evaluation complete.")


