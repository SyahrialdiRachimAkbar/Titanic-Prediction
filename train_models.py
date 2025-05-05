#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load preprocessed data
processed_data_path = "/home/ubuntu/data_analysis_project/processed_data.csv"
df_processed = pd.read_csv(processed_data_path)

# Separate features (X) and target (y)
X = df_processed.drop("Survived", axis=1)
y = df_processed["Survived"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# --- Model Training --- 

models = {
    "Logistic Regression": LogisticRegression(solver="liblinear", random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

trained_models = {}
model_dir = "/home/ubuntu/data_analysis_project/models"
import os
os.makedirs(model_dir, exist_ok=True)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Save the trained model
    model_path = os.path.join(model_dir, f'{name.replace(" ", "_").lower()}.joblib')
    joblib.dump(model, model_path)
    trained_models[name] = model
    print(f"{name} model saved to {model_path}")

print("\nModel training complete.")

# --- Optional: Hyperparameter Tuning (Example for Random Forest) --- 
# This can take longer, uncomment if needed and adjust parameters

# print("\nStarting Hyperparameter Tuning for Random Forest...")
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# 
# rf = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# 
# print(f"Best Parameters for Random Forest: {grid_search.best_params_}")
# best_rf_model = grid_search.best_estimator_
# y_pred_best_rf = best_rf_model.predict(X_test)
# best_rf_accuracy = accuracy_score(y_test, y_pred_best_rf)
# print(f"Tuned Random Forest Accuracy: {best_rf_accuracy:.4f}")
# 
# # Save the tuned model
# tuned_model_path = os.path.join(model_dir, "random_forest_tuned.joblib")
# joblib.dump(best_rf_model, tuned_model_path)
# print(f"Tuned Random Forest model saved to {tuned_model_path}")


