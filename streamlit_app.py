#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt # Keep for feature importance plot
import seaborn as sns # Keep for feature importance plot
import plotly.express as px
import plotly.graph_objects as go

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
DATA_PATH = os.path.join("Titanic-Dataset.csv")
PREPROCESSOR_PATH = os.path.join("preprocessor.joblib")
MODEL_PATH = os.path.join("models", "random_forest.joblib") # Using Random Forest as the best model
EVALUATION_METRICS_PATH = os.path.join("evaluation", "evaluation_metrics.csv")
VISUALIZATIONS_DIR = os.path.join("visualizations") # Keep for static images if needed
EVALUATION_DIR = os.path.join("evaluation")
PROCESSED_DATA_PATH = os.path.join("processed_data.csv") # Load processed data for EDA

# Load necessary files
df_raw = load_data(DATA_PATH)
preprocessor = load_model(PREPROCESSOR_PATH)
model = load_model(MODEL_PATH)
evaluation_metrics = load_data(EVALUATION_METRICS_PATH)

# Load processed data for interactive plots (handle potential errors)
try:
    df_processed_viz = load_data(PROCESSED_DATA_PATH)
    # Get original columns before one-hot encoding for easier plotting
    # Need to reconstruct or load the dataframe *before* one-hot encoding for some plots
    # Let's reload the raw data and apply only the necessary imputation/feature engineering steps for viz
    df_viz = df_raw.copy()
    median_ages = df_viz.groupby(["Pclass", "Sex"])["Age"].transform("median")
    df_viz["Age"] = df_viz["Age"].fillna(median_ages)
    most_frequent_embarked = df_viz["Embarked"].mode()[0]
    df_viz["Embarked"] = df_viz["Embarked"].fillna(most_frequent_embarked)
    df_viz["FamilySize"] = df_viz["SibSp"] + df_viz["Parch"] + 1
except FileNotFoundError:
    st.error(f"Processed data file not found at {PROCESSED_DATA_PATH}. Cannot display interactive EDA.")
    df_viz = None # Set df_viz to None if loading fails
except Exception as e:
    st.error(f"Error loading data for visualization: {e}")
    df_viz = None

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
    *   **Exploratory Data Analysis:** Visualize patterns and insights from the data interactively.
    *   **Model Performance:** See how well different machine learning models performed.
    *   **Make a Prediction:** Input passenger details to predict their survival chances using the best-performing model.
    """)
    st.subheader("Dataset Preview")
    st.dataframe(df_raw.head())

elif page == "Exploratory Data Analysis":
    st.title("📊 Interactive Exploratory Data Analysis (EDA)")
    st.markdown("Visualizing the Titanic dataset helps us understand the relationships between different features and survival. Hover over the plots for details, zoom, and pan.")

    if df_viz is not None:
        # Plot 1: Survival Distribution
        st.subheader("Survival Distribution")
        survival_counts = df_viz["Survived"].value_counts().reset_index()
        survival_counts.columns = ["Survived", "Count"]
        survival_counts["Survived"] = survival_counts["Survived"].map({0: "No", 1: "Yes"})
        fig_survival = px.bar(survival_counts, x="Survived", y="Count", color="Survived", 
                              title="Survival Distribution (0 = No, 1 = Yes)", 
                              labels={"Count": "Number of Passengers"},
                              color_discrete_map={"No": "#440154", "Yes": "#21918c"})
        st.plotly_chart(fig_survival, use_container_width=True)

        # Plot 2: Survival by Sex
        st.subheader("Survival Distribution by Sex")
        fig_sex = px.histogram(df_viz, x="Sex", color="Survived", barmode="group",
                               title="Survival Distribution by Sex",
                               labels={"Survived": "Survived (0=No, 1=Yes)"},
                               category_orders={"Survived": [0, 1]})
        fig_sex.update_layout(yaxis_title="Number of Passengers")
        st.plotly_chart(fig_sex, use_container_width=True)

        # Plot 3: Survival by Pclass
        st.subheader("Survival Distribution by Pclass")
        fig_pclass = px.histogram(df_viz, x="Pclass", color="Survived", barmode="group",
                                  title="Survival Distribution by Passenger Class",
                                  labels={"Pclass": "Passenger Class", "Survived": "Survived (0=No, 1=Yes)"},
                                  category_orders={"Survived": [0, 1]})
        fig_pclass.update_layout(yaxis_title="Number of Passengers", xaxis=dict(tickmode=\'linear\'))
        st.plotly_chart(fig_pclass, use_container_width=True)

        # Plot 4: Age Distribution
        st.subheader("Age Distribution")
        fig_age = px.histogram(df_viz, x="Age", nbins=30, title="Age Distribution", marginal="box")
        fig_age.update_layout(yaxis_title="Number of Passengers")
        st.plotly_chart(fig_age, use_container_width=True)

        # Plot 5: Age Distribution by Survival
        st.subheader("Age Distribution by Survival")
        fig_age_surv = px.histogram(df_viz, x="Age", color="Survived", nbins=30, 
                                    title="Age Distribution by Survival", marginal="box",
                                    labels={"Survived": "Survived (0=No, 1=Yes)"},
                                    category_orders={"Survived": [0, 1]})
        fig_age_surv.update_layout(yaxis_title="Number of Passengers")
        st.plotly_chart(fig_age_surv, use_container_width=True)

        # Plot 6: Fare Distribution
        st.subheader("Fare Distribution")
        fig_fare = px.histogram(df_viz, x="Fare", nbins=50, title="Fare Distribution (Limited to Fare < 300)", marginal="box")
        fig_fare.update_layout(yaxis_title="Number of Passengers", xaxis_range=[0,300]) # Limit x-axis
        st.plotly_chart(fig_fare, use_container_width=True)

        # Plot 7: Family Size Distribution
        st.subheader("Family Size Distribution")
        fig_fam_size = px.histogram(df_viz, x="FamilySize", title="Family Size Distribution")
        fig_fam_size.update_layout(yaxis_title="Number of Passengers", xaxis_title="Family Size (including self)", xaxis=dict(tickmode=\'linear\'))
        st.plotly_chart(fig_fam_size, use_container_width=True)

        # Plot 8: Survival by Family Size
        st.subheader("Survival Distribution by Family Size")
        fig_fam_surv = px.histogram(df_viz, x="FamilySize", color="Survived", barmode="group",
                                    title="Survival Distribution by Family Size",
                                    labels={"Survived": "Survived (0=No, 1=Yes)"},
                                    category_orders={"Survived": [0, 1]})
        fig_fam_surv.update_layout(yaxis_title="Number of Passengers", xaxis_title="Family Size (including self)", xaxis=dict(tickmode=\'linear\'))
        st.plotly_chart(fig_fam_surv, use_container_width=True)

        # Plot 9: Survival by Embarked
        st.subheader("Survival Distribution by Embarked")
        fig_embarked = px.histogram(df_viz, x="Embarked", color="Survived", barmode="group",
                                    title="Survival Distribution by Port of Embarkation",
                                    labels={"Embarked": "Port (S=Southampton, C=Cherbourg, Q=Queenstown)", "Survived": "Survived (0=No, 1=Yes)"},
                                    category_orders={"Survived": [0, 1]})
        fig_embarked.update_layout(yaxis_title="Number of Passengers")
        st.plotly_chart(fig_embarked, use_container_width=True)

        # Plot 10: Correlation Heatmap (Numerical Features Only)
        st.subheader("Correlation Heatmap")
        numeric_cols_viz = df_viz.select_dtypes(include=np.number).columns
        # Drop PassengerId for correlation heatmap
        numeric_cols_viz = numeric_cols_viz.drop("PassengerId", errors=\'ignore\')
        corr = df_viz[numeric_cols_viz].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", 
                             title="Correlation Heatmap of Numerical Features",
                             color_continuous_scale=\'viridis\')
        st.plotly_chart(fig_corr, use_container_width=True)

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
    else:
        st.warning("Could not load data to display visualizations.")


elif page == "Model Performance":
    st.title("🤖 Model Performance Evaluation")
    st.markdown("Several machine learning models were trained to predict survival. Here\'s how they performed on the test data:")

    st.subheader("Performance Metrics")
    # Rename the first column for clarity if it exists
    if "" in evaluation_metrics.columns:
         evaluation_metrics.rename(columns={"": "Model"}, inplace=True)
    elif "Unnamed: 0" in evaluation_metrics.columns:
         evaluation_metrics.rename(columns={"Unnamed: 0": "Model"}, inplace=True)
    
    # Check if 'Model' column exists before setting index
    if "Model" in evaluation_metrics.columns:
        st.dataframe(evaluation_metrics.set_index("Model"))
    else:
        st.dataframe(evaluation_metrics)
        st.warning("Could not identify Model column in evaluation metrics file.")
        
    st.markdown("**Metrics:** Accuracy, Precision, Recall, F1-Score, ROC AUC. Higher values are generally better.")

    st.subheader("Confusion Matrices")
    st.markdown("These show the number of correct and incorrect predictions for each class (Survived vs. Not Survived).")
    
    model_names_cm = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    cols_cm = st.columns(len(model_names_cm))
    for i, model_name in enumerate(model_names_cm):
        cm_path = os.path.join(EVALUATION_DIR, f'{model_name.replace(" ", "_").lower()}_confusion_matrix.png')
        if os.path.exists(cm_path):
            with cols_cm[i]:
                st.image(cm_path, caption=f"{model_name} Confusion Matrix", use_container_width=True)
        else:
             with cols_cm[i]:
                st.warning(f"CM for {model_name} not found.")

    st.subheader("ROC Curves")
    st.markdown("The Receiver Operating Characteristic (ROC) curve shows the trade-off between true positive rate and false positive rate. A curve closer to the top-left corner indicates better performance.")
    roc_path = os.path.join(EVALUATION_DIR, "combined_roc_curve.png")
    if os.path.exists(roc_path):
        st.image(roc_path, caption="ROC Curves for All Models", use_container_width=True)
    else:
        st.warning("Combined ROC curve image not found.")

    st.subheader("Conclusion")
    st.markdown("Based on the evaluation metrics (particularly Accuracy and F1-Score), the **Random Forest** model was selected as the best model for the prediction task in this application.")

elif page == "Make a Prediction":
    st.title("🔮 Make a Survival Prediction")
    st.markdown("Enter the passenger\'s details below to predict their survival chances using the trained Random Forest model.")

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
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    importances = model.feature_importances_
                    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
                    
                    # Plotting feature importance using Plotly
                    fig_importance = px.bar(importance_df.head(10), x='importance', y='feature', orientation='h',
                                            title='Top 10 Feature Importances for the Prediction',
                                            labels={'importance': 'Importance Score', 'feature': 'Feature'})
                    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_importance, use_container_width=True)

                except Exception as plot_err:
                    st.warning(f"Could not plot feature importance: {plot_err}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure all inputs are valid.")

