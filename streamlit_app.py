#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Assets ---
@st.cache_data # Cache data loading
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource # Cache model and preprocessor loading
def load_model(path):
    return joblib.load(path)

# Define paths
PROJECT_DIR = "/home/ubuntu/data_analysis_project"
DATA_PATH = os.path.join(PROJECT_DIR, "Titanic-Dataset.csv")
PREPROCESSOR_PATH = os.path.join(PROJECT_DIR, "preprocessor.joblib")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "random_forest.joblib") # Using Random Forest as the best model
EVALUATION_METRICS_PATH = os.path.join(PROJECT_DIR, "evaluation", "evaluation_metrics.csv")
VISUALIZATIONS_DIR = os.path.join(PROJECT_DIR, "visualizations")
EVALUATION_DIR = os.path.join(PROJECT_DIR, "evaluation")

# Load necessary files
df_raw = load_data(DATA_PATH)
preprocessor = load_model(PREPROCESSOR_PATH)
model = load_model(MODEL_PATH)
evaluation_metrics = load_data(EVALUATION_METRICS_PATH)

# --- Sidebar --- 
st.sidebar.title("🚢 Titanic Survival Prediction")
st.sidebar.markdown("Explore the analysis of the Titanic dataset and predict passenger survival.")

page = st.sidebar.radio("Navigate", ["Introduction", "Exploratory Data Analysis", "Model Performance", "Make a Prediction"], index=0)

st.sidebar.markdown("--- ")
st.sidebar.info("This app demonstrates data analysis and machine learning on the Titanic dataset.")

# --- Main Page Content ---

if page == "Introduction":
    st.title("Welcome to the Titanic Survival Analysis App")
    st.markdown("""
    This application explores the famous Titanic dataset, which contains information about passengers aboard the RMS Titanic, 
    including whether they survived the tragic sinking on April 15, 1912.

    **Goal:** To analyze the factors influencing survival and build a machine learning model to predict if a passenger would survive based on their characteristics.

    **Dataset:** The dataset includes features like passenger class (Pclass), sex, age, fare paid, port of embarkation (Embarked), and family size.

    Use the sidebar to navigate through the different sections:
    *   **Exploratory Data Analysis:** Visualize patterns and insights from the data.
    *   **Model Performance:** See how well different machine learning models performed.
    *   **Make a Prediction:** Input passenger details to predict their survival chances using the best-performing model.
    """)
    st.subheader("Dataset Preview")
    st.dataframe(df_raw.head())

elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Visualizing the Titanic dataset helps us understand the relationships between different features and survival.")

    viz_files = {
        "Survival Distribution": "survival_distribution.png",
        "Survival by Sex": "survival_by_sex.png",
        "Survival by Pclass": "survival_by_pclass.png",
        "Age Distribution": "age_distribution.png",
        "Age Distribution by Survival": "age_distribution_by_survival.png",
        "Fare Distribution": "fare_distribution.png",
        "Family Size Distribution": "familysize_distribution.png",
        "Survival by Family Size": "survival_by_familysize.png",
        "Survival by Embarked": "survival_by_embarked.png",
        "Correlation Heatmap": "correlation_heatmap.png"
    }

    for title, fname in viz_files.items():
        img_path = os.path.join(VISUALIZATIONS_DIR, fname)
        if os.path.exists(img_path):
            st.subheader(title)
            st.image(img_path, use_column_width=True)
        else:
            st.warning(f"Visualization '{title}' not found at {img_path}")

    st.subheader("Key EDA Findings")
    st.markdown("""
    *   **Survival Rate:** Fewer than 40% of passengers in this dataset survived.
    *   **Sex:** Females had a significantly higher survival rate than males.
    *   **Pclass:** Passengers in 1st class had a much higher survival rate compared to 2nd and 3rd class passengers.
    *   **Age:** Younger passengers (children) had a higher chance of survival. There were many passengers in the 20-40 age range.
    *   **Fare:** Passengers who paid higher fares generally had better survival rates, likely correlated with Pclass.
    *   **Family Size:** Passengers traveling alone or with very large families had lower survival rates compared to those with small families (2-4 members).
    *   **Embarked:** Passengers embarking from Cherbourg (C) had a slightly higher survival rate.
    """)

elif page == "Model Performance":
    st.title("🤖 Model Performance Evaluation")
    st.markdown("Several machine learning models were trained to predict survival. Here's how they performed on the test data:")

    st.subheader("Performance Metrics")
    # Rename the first column for clarity
    evaluation_metrics.rename(columns={"": "Model"}, inplace=True)
    st.dataframe(evaluation_metrics.set_index("Model"))
    st.markdown("**Metrics:** Accuracy, Precision, Recall, F1-Score, ROC AUC. Higher values are generally better.")

    st.subheader("Confusion Matrices")
    st.markdown("These show the number of correct and incorrect predictions for each class (Survived vs. Not Survived).")
    
    model_names_cm = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    cols_cm = st.columns(len(model_names_cm))
    for i, model_name in enumerate(model_names_cm):
        cm_path = os.path.join(EVALUATION_DIR, f'{model_name.replace(" ", "_").lower()}_confusion_matrix.png')
        if os.path.exists(cm_path):
            with cols_cm[i]:
                st.image(cm_path, caption=f"{model_name} Confusion Matrix", use_column_width=True)
        else:
             with cols_cm[i]:
                st.warning(f"CM for {model_name} not found.")

    st.subheader("ROC Curves")
    st.markdown("The Receiver Operating Characteristic (ROC) curve shows the trade-off between true positive rate and false positive rate. A curve closer to the top-left corner indicates better performance.")
    roc_path = os.path.join(EVALUATION_DIR, "combined_roc_curve.png")
    if os.path.exists(roc_path):
        st.image(roc_path, caption="ROC Curves for All Models", use_column_width=True)
    else:
        st.warning("Combined ROC curve image not found.")

    st.subheader("Conclusion")
    st.markdown("Based on the evaluation metrics (particularly Accuracy and F1-Score), the **Random Forest** model was selected as the best model for the prediction task in this application.")

elif page == "Make a Prediction":
    st.title("🔮 Make a Survival Prediction")
    st.markdown("Enter the passenger's details below to predict their survival chances using the trained Random Forest model.")

    # Input form
    with st.form("prediction_form"):
        st.subheader("Passenger Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=0)
            sex = st.radio("Sex", ["male", "female"], index=1)
            age = st.number_input("Age", min_value=0.1, max_value=100.0, value=30.0, step=0.5)
        
        with col2:
            fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=30.0, step=1.0)
            embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], index=0, help="S=Southampton, C=Cherbourg, Q=Queenstown")
            family_size = st.number_input("Family Size (including self)", min_value=1, max_value=15, value=1, step=1)

        submitted = st.form_submit_button("Predict Survival")

    if submitted:
        # Create input DataFrame for preprocessing
        input_data = pd.DataFrame({
            "Pclass": [pclass],
            "Sex": [sex],
            "Age": [age],
            "Fare": [fare],
            "Embarked": [embarked],
            "FamilySize": [family_size]
        })
        
        st.subheader("Input Data Summary")
        st.write(input_data)

        try:
            # Preprocess the input data
            input_processed = preprocessor.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_processed)[0]
            prediction_proba = model.predict_proba(input_processed)[0]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(f"**Predicted Outcome: Survived** (Probability: {prediction_proba[1]:.2f})")
                st.balloons()
            else:
                st.error(f"**Predicted Outcome: Did Not Survive** (Probability: {prediction_proba[0]:.2f})")
                
            # Show feature importance (if available for Random Forest)
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance (How features influenced the prediction)")
                # Get feature names after preprocessing
                feature_names = preprocessor.get_feature_names_out()
                importances = model.feature_importances_
                importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
                
                # Plotting feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=importance_df.head(10), palette='viridis', ax=ax) # Show top 10
                ax.set_title('Top 10 Feature Importances for the Prediction')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure all inputs are valid.")


