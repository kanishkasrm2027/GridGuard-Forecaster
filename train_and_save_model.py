import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# -------------------- Step 1: Load the Dataset from CSV --------------------
try:
    # Load the dataset from the CSV file
    df = pd.read_csv('projects.csv')
    print("Dataset loaded successfully from projects.csv")
except FileNotFoundError:
    print("Error: The file 'projects.csv' was not found.")
    print("Please make sure the file is in the same directory as this script.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# -------------------- Step 2: Preprocessing and Feature Engineering --------------------
# Define features (X) and target variables (y)
features = [
    'Project_Type', 'Planned_Duration_Days', 'Planned_Cost_INR_Cr', 'Terrain_Factor',
    'Regulatory_Approval_Days', 'Historical_Delay_Reason', 'Weather_Impact_Score',
    'Vendor_Performance_Rating', 'Material_Availability_Issue',
    'Demand_Supply_Scenario', 'Manpower_Availability_Score'
]

X = df[features]
y_time = df['Time_Overrun_Days']
y_cost = df['Cost_Overrun_INR_Cr']

# Define categorical and numerical features for preprocessing
categorical_features = [
    'Project_Type', 'Terrain_Factor', 'Historical_Delay_Reason',
    'Vendor_Performance_Rating', 'Demand_Supply_Scenario'
]
numerical_features = [
    'Planned_Duration_Days', 'Planned_Cost_INR_Cr',
    'Regulatory_Approval_Days', 'Weather_Impact_Score',
    'Manpower_Availability_Score', 'Material_Availability_Issue'
]

# Create a preprocessing pipeline for both categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# -------------------- Step 3: Ensemble Machine Learning Models (XGBoost) --------------------
# Split data into training and testing sets
X_train, X_test, y_time_train, y_time_test = train_test_split(
    X, y_time, test_size=0.2, random_state=42
)
_, _, y_cost_train, y_cost_test = train_test_split(
    X, y_cost, test_size=0.2, random_state=42
)

# Create a full pipeline for the time overrun model
time_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Create a full pipeline for the cost overrun model
cost_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Train the models
print("Training Time Overrun model...")
time_model.fit(X_train, y_time_train)
print("Training Cost Overrun model...")
cost_model.fit(X_train, y_cost_train)

# -------------------- Step 4: Model Evaluation --------------------
# Make predictions on the test set
y_time_pred = time_model.predict(X_test)
y_cost_pred = cost_model.predict(X_test)

# Calculate evaluation metrics
mae_time = mean_absolute_error(y_time_test, y_time_pred)
r2_time = r2_score(y_time_test, y_time_pred)
mae_cost = mean_absolute_error(y_cost_test, y_cost_pred)
r2_cost = r2_score(y_cost_test, y_cost_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Time Overrun Model: MAE = {mae_time:.2f}, R-squared = {r2_time:.2f}")
print(f"Cost Overrun Model: MAE = {mae_cost:.2f}, R-squared = {r2_cost:.2f}")

# -------------------- Step 5: Save the Models and Metrics --------------------
print("\n--- Saving Models and Metrics ---")
joblib.dump(time_model, 'time_model.joblib')
joblib.dump(cost_model, 'cost_model.joblib')
joblib.dump(X_test, 'X_test.joblib')
joblib.dump(y_time_test, 'y_time_test.joblib')
joblib.dump(y_cost_test, 'y_cost_test.joblib')
joblib.dump(mae_time, 'mae_time.joblib')
joblib.dump(r2_time, 'r2_time.joblib')
joblib.dump(mae_cost, 'mae_cost.joblib')
joblib.dump(r2_cost, 'r2_cost.joblib')

print("Models and metrics have been saved successfully.")
print("You can now run 'streamlit run app.py' to launch the dashboard.")
