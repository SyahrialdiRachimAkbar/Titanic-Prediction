#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
df = pd.read_csv("/home/ubuntu/data_analysis_project/Titanic-Dataset.csv")

# --- 1. Handle Missing Values --- 

# Impute Age: Use median age grouped by Pclass and Sex
# Calculate median ages
median_ages = df.groupby(["Pclass", "Sex"])["Age"].transform("median")
# Fill missing ages
df["Age"] = df["Age"].fillna(median_ages)

# Impute Embarked: Use the mode (most frequent value)
most_frequent_embarked = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(most_frequent_embarked)

# Drop Cabin: Too many missing values
df = df.drop("Cabin", axis=1)

# Verify no more missing values in Age and Embarked
print("--- Missing Values After Imputation ---")
print(df.isnull().sum())

# --- 2. Feature Engineering --- 

# Create FamilySize
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Drop original SibSp, Parch, Name, Ticket, PassengerId (less useful for direct modeling)
df = df.drop(["SibSp", "Parch", "Name", "Ticket", "PassengerId"], axis=1)

print("\n--- DataFrame Head After Feature Engineering ---")
print(df.head())

# --- 3. Data Visualization --- 

output_dir = "/home/ubuntu/data_analysis_project/visualizations"
import os
os.makedirs(output_dir, exist_ok=True)

# Set style
sns.set(style="whitegrid")

# Distribution of Survival
plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", data=df, palette="viridis")
plt.title("Survival Distribution (0 = No, 1 = Yes)")
plt.savefig(os.path.join(output_dir, "survival_distribution.png"))
plt.close()

# Survival by Sex
plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", hue="Sex", data=df, palette="viridis")
plt.title("Survival Distribution by Sex")
plt.savefig(os.path.join(output_dir, "survival_by_sex.png"))
plt.close()

# Survival by Pclass
plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", hue="Pclass", data=df, palette="viridis")
plt.title("Survival Distribution by Pclass")
plt.savefig(os.path.join(output_dir, "survival_by_pclass.png"))
plt.close()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["Age"], kde=True, bins=30)
plt.title("Age Distribution")
plt.savefig(os.path.join(output_dir, "age_distribution.png"))
plt.close()

# Age Distribution by Survival
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30, palette="viridis")
plt.title("Age Distribution by Survival")
plt.savefig(os.path.join(output_dir, "age_distribution_by_survival.png"))
plt.close()

# Fare Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["Fare"], kde=True, bins=40)
plt.title("Fare Distribution")
plt.xlim(0, 300) # Limit x-axis for better visibility
plt.savefig(os.path.join(output_dir, "fare_distribution.png"))
plt.close()

# FamilySize Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="FamilySize", data=df, palette="viridis")
plt.title("Family Size Distribution")
plt.savefig(os.path.join(output_dir, "familysize_distribution.png"))
plt.close()

# Survival by FamilySize
plt.figure(figsize=(10, 6))
sns.countplot(x="FamilySize", hue="Survived", data=df, palette="viridis")
plt.title("Survival Distribution by Family Size")
plt.savefig(os.path.join(output_dir, "survival_by_familysize.png"))
plt.close()

# Survival by Embarked
plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", hue="Embarked", data=df, palette="viridis")
plt.title("Survival Distribution by Embarked")
plt.savefig(os.path.join(output_dir, "survival_by_embarked.png"))
plt.close()

# Correlation Heatmap (Numerical Features Only)
numeric_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="viridis", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

print(f"\nVisualizations saved to {output_dir}")

# --- 4. Data Preprocessing --- 

# Separate features (X) and target (y)
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ("scaler", StandardScaler()) # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore")) # One-hot encode categorical features
])

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="passthrough" # Keep other columns (if any), though we dropped most
)

# Apply preprocessing to the features
X_processed = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding
feature_names_out = preprocessor.get_feature_names_out()

# Convert processed data back to DataFrame (optional, for inspection)
X_processed_df = pd.DataFrame(X_processed, columns=feature_names_out)
print("\n--- Processed Features Head ---")
print(X_processed_df.head())

# Save the preprocessed data and the preprocessor
processed_data_path = "/home/ubuntu/data_analysis_project/processed_data.csv"
X_processed_df["Survived"] = y.values # Add target back for saving
X_processed_df.to_csv(processed_data_path, index=False)

preprocessor_path = "/home/ubuntu/data_analysis_project/preprocessor.joblib"
joblib.dump(preprocessor, preprocessor_path)

print(f"\nProcessed data saved to {processed_data_path}")
print(f"Preprocessor saved to {preprocessor_path}")


