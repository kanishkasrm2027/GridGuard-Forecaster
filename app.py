import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import toml

st.set_page_config(layout="wide")

# --- Function to load saved models and metrics ---
@st.cache_data
def load_saved_models():
    """
    Loads the pre-trained models and their evaluation metrics from disk.
    This function is cached to run only once.
    """
    try:
        time_model = joblib.load('time_model.joblib')
        cost_model = joblib.load('cost_model.joblib')
        X_test = joblib.load('X_test.joblib')
        y_time_test = joblib.load('y_time_test.joblib')
        y_cost_test = joblib.load('y_cost_test.joblib')
        mae_time = joblib.load('mae_time.joblib')
        r2_time = joblib.load('r2_time.joblib')
        mae_cost = joblib.load('mae_cost.joblib')
        r2_cost = joblib.load('r2_cost.joblib')
        return time_model, cost_model, X_test, y_time_test, y_cost_test, mae_time, r2_time, mae_cost, r2_cost
    except FileNotFoundError:
        st.error("Model files not found. Please run the `train_and_save_model.py` script first.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the models: {e}")
        st.stop()

# Load models and metrics
time_model, cost_model, X_test, y_time_test, y_cost_test, mae_time, r2_time, mae_cost, r2_cost = load_saved_models()

# --- Dashboard Layout and UI ---
st.title("GridGuard Forecaster")
st.markdown("Predict and analyze the potential time and cost overruns for new projects.")

# --- Sidebar for User Input ---
st.sidebar.header("New Project Details")
st.sidebar.markdown("Enter the characteristics of a new project to get a prediction.")

project_type = st.sidebar.selectbox("Project Type", ['Substation', 'Overhead Line', 'Underground Cable'])
planned_duration = st.sidebar.number_input("Planned Duration (Days)", min_value=100, max_value=1500, value=800)
planned_cost = st.sidebar.number_input("Planned Cost (INR Cr)", min_value=10.0, max_value=500.0, value=175.0, step=0.5)
terrain_factor = st.sidebar.selectbox("Terrain Factor", ['Plain', 'Hilly', 'Forest', 'Desert'])
regulatory_days = st.sidebar.number_input("Regulatory Approval (Days)", min_value=50, max_value=300, value=180)
delay_reason = st.sidebar.selectbox("Historical Delay Reason", ['None', 'Land Acquisition', 'Environmental Clearance', 'Weather', 'Supply Chain', 'Vendor Issues', 'Manpower Shortage'])
weather_score = st.sidebar.slider("Weather Impact Score (1-5)", 1, 5, 4)
vendor_rating = st.sidebar.selectbox("Vendor Performance Rating", ['Excellent', 'Good', 'Average', 'Poor'])
material_issue = st.sidebar.checkbox("Material Availability Issue?")
demand_scenario = st.sidebar.selectbox("Demand-Supply Scenario", ['Low Demand', 'Balanced', 'High Demand'])
manpower_score = st.sidebar.slider("Manpower Availability Score (1-5)", 1, 5, 2)

predict_button = st.sidebar.button("Predict Overruns")

# --- Main Content Area ---
if predict_button:
    new_project = {
        'Project_Type': project_type,
        'Planned_Duration_Days': planned_duration,
        'Planned_Cost_INR_Cr': planned_cost,
        'Terrain_Factor': terrain_factor,
        'Regulatory_Approval_Days': regulatory_days,
        'Historical_Delay_Reason': delay_reason,
        'Weather_Impact_Score': weather_score,
        'Vendor_Performance_Rating': vendor_rating,
        'Material_Availability_Issue': material_issue,
        'Demand_Supply_Scenario': demand_scenario,
        'Manpower_Availability_Score': manpower_score
    }
    
    new_project_df = pd.DataFrame([new_project])
    
    predicted_time_overrun = time_model.predict(new_project_df)[0]
    predicted_cost_overrun = cost_model.predict(new_project_df)[0]
    
    st.header("Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Time Overrun", value=f"{predicted_time_overrun:.2f} Days")
    with col2:
        st.metric(label="Predicted Cost Overrun", value=f"₹ {predicted_cost_overrun:.2f} Cr")
    
    st.success("The model has generated a prediction for the new project based on its characteristics.")

# --- Model Evaluation Section ---
st.header("Model Evaluation Summary")
st.markdown("These metrics indicate the overall performance of the models on unseen data.")

col3, col4, col5, col6 = st.columns(4)

with col3:
    st.subheader("Time Model")
    st.markdown(f"**Mean Absolute Error (MAE):** `{mae_time:.2f}`")
    st.markdown(f"**R-squared ($R^2$):** `{r2_time:.2f}`")
with col4:
    st.subheader("Cost Model")
    st.markdown(f"**Mean Absolute Error (MAE):** `{mae_cost:.2f}`")
    st.markdown(f"**R-squared ($R^2$):** `{r2_cost:.2f}`")

# --- Evaluation Graphs ---
st.subheader("Predicted vs. Actual Values")
st.markdown("These plots show how closely the model's predictions align with the actual overruns in the test data.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.regplot(x=y_time_test, y=time_model.predict(X_test), ax=ax1, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
ax1.set_title("Time Overrun: Predicted vs. Actual")
ax1.set_xlabel("Actual Time Overrun (Days)")
ax1.set_ylabel("Predicted Time Overrun (Days)")
ax1.set_aspect('equal', adjustable='box')
ax1.plot([y_time_test.min(), y_time_test.max()], [y_time_test.min(), y_time_test.max()], 'k--', lw=2, alpha=0.5)

sns.regplot(x=y_cost_test, y=cost_model.predict(X_test), ax=ax2, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
ax2.set_title("Cost Overrun: Predicted vs. Actual")
ax2.set_xlabel("Actual Cost Overrun (INR Cr)")
ax2.set_ylabel("Predicted Cost Overrun (INR Cr)")
ax2.set_aspect('equal', adjustable='box')
ax2.plot([y_cost_test.min(), y_cost_test.max()], [y_cost_test.min(), y_cost_test.max()], 'k--', lw=2, alpha=0.5)

st.pyplot(fig)

# --- Risk Hotspot Analysis (SHAP) ---
st.header("Risk Hotspot Analysis (SHAP)")
st.info("This analysis reveals the most influential factors impacting the model's predictions. The plots are generated from a sample of the test data and represent the model's general learned behavior, not a single prediction.")
st.markdown("A **positive SHAP value** (shown in red) means a feature's value pushes the prediction **up** (i.e., a larger overrun). A **negative SHAP value** (shown in blue) means it pushes the prediction **down** (i.e., a smaller overrun).")

# Get a random sample from the test set for SHAP analysis to make it faster
sample_size = min(len(X_test), 50)  # Use a maximum of 50 samples
sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
X_test_sample = X_test.iloc[sample_indices]

# Explain the Time Overrun model
explainer_time = shap.TreeExplainer(time_model.named_steps['regressor'])
shap_values_time = explainer_time.shap_values(time_model.named_steps['preprocessor'].transform(X_test_sample))

# Explain the Cost Overrun model
explainer_cost = shap.TreeExplainer(cost_model.named_steps['regressor'])
shap_values_cost = explainer_cost.shap_values(cost_model.named_steps['preprocessor'].transform(X_test_sample))

# Get feature names from the preprocessor
feature_names = (
    time_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(time_model.named_steps['preprocessor'].transformers_[0][2]).tolist() +
    time_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(time_model.named_steps['preprocessor'].transformers_[1][2]).tolist()
)

st.subheader("Time Overrun Feature Importance")
fig_shap_time = plt.figure()
shap.summary_plot(shap_values_time, features=time_model.named_steps['preprocessor'].transform(X_test_sample), feature_names=feature_names, show=False)
st.pyplot(fig_shap_time)
plt.clf()

st.subheader("Cost Overrun Feature Importance")
fig_shap_cost = plt.figure()
shap.summary_plot(shap_values_cost, features=cost_model.named_steps['preprocessor'].transform(X_test_sample), feature_names=feature_names, show=False)
st.pyplot(fig_shap_cost)
plt.clf()

# --- Technologies Used ---
st.header("Technologies Used")
st.markdown("""
This application is built using a stack of open-source Python libraries, each serving a specific purpose in the predictive pipeline.

**Core Frameworks**
- **Streamlit:** The main framework used to build and deploy the interactive web dashboard. It turns Python scripts into shareable web applications with minimal effort.
- **Scikit-learn:** A foundational library for machine learning, used for data preprocessing and creating the predictive pipeline. It provides tools for data scaling, encoding, and model chaining.
- **XGBoost:** The high-performance, tree-based algorithm used for the predictive models. It is an ensemble method known for its speed and accuracy in solving regression and classification problems.

**Data & File Management**
- **Pandas:** Used for efficient data handling and manipulation of the project dataset. It provides powerful data structures like DataFrames, making data cleaning and preparation straightforward.
- **Joblib:** The library used to save and load the trained models and evaluation metrics to and from disk. This allows us to separate the model training process from the live application, improving performance.
- **Toml:** A configuration file format used to programmatically set the Streamlit dashboard's visual theme.

**Visualization & Explainability**
- **Matplotlib / Seaborn:** Used to create the evaluation plots (e.g., predicted vs. actual values). These libraries provide robust tools for generating high-quality statistical graphs.
- **SHAP:** A key library from Explainable AI (XAI) that provides the visual risk "hotspot" analysis. It explains model predictions by showing the contribution of each feature, making the model's decisions transparent.
""")

# --- Glossary Section ---
with st.expander("Glossary and Technical Terms"):
    st.markdown("""
    **Machine Learning Model:** An algorithm trained on data to recognize patterns and make predictions or decisions. We used XGBoost, an advanced tree-based model.

    **MAE (Mean Absolute Error):** The average difference between the predicted value and the actual value. A lower MAE indicates a more accurate model.

    **R-squared ($R^2$):** A statistical measure that represents the proportion of the variance for a dependent variable that is explained by the independent variables. A higher $R^2$ value (closer to 1) indicates a better fit of the model to the data.

    **Feature Engineering:** The process of using domain knowledge to create new features from raw data to improve model performance.

    **One-Hot Encoding:** A technique to convert categorical data into a numerical format. Each category is turned into a new binary column.

    **StandardScaler:** A preprocessing technique to standardize numerical features by removing the mean and scaling to unit variance. This prevents features with larger scales from dominating the model.
    """)
