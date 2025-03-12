import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and scaler
with open('model_xgb.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Title
st.title("Employee Attrition Prediction App")
st.write("Enter the employee details below to predict whether the employee will leave the company.")

# Input fields
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
last_performance_rating = st.selectbox("Last Performance Rating", options=[0, 1, 2, 3])
job_title = st.selectbox("Job Title", options=[0, 1, 2, 3, 4, 5, 6])
dept_name = st.selectbox("Department", options=[0, 1, 2, 3, 4, 5, 6])
no_of_projects = st.number_input("Number of Projects", min_value=1, max_value=20, value=3)
salary = st.number_input("Salary", min_value=1000, max_value=100000, value=50000)

# Prepare the input data
input_data = pd.DataFrame(
    {
        "sex": [sex],
        "last_performance_rating": [last_performance_rating],
        "job_title": [job_title],
        "dept_name": [dept_name],
        "no_of_projects": [no_of_projects],
        "salary": [salary],
    }
)

# Scale numeric features
input_data[["no_of_projects", "salary"]] = scaler.transform(input_data[["no_of_projects", "salary"]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[:, 1][0]
    
    if prediction == 1:
        st.error(f"🚨 The employee is likely to leave the company (Confidence: {prediction_proba:.2f})")
    else:
        st.success(f"✅ The employee is likely to stay with the company (Confidence: {prediction_proba:.2f})")

# Feature importance plot
if st.checkbox("Show Feature Importance"):
    feature_importance = model.feature_importances_
    feature_names = input_data.columns
    
    st.write("### Feature Importance")
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    
    st.bar_chart(importance_df.set_index("Feature"))

# Prediction probability plot
if st.checkbox("Show Prediction Probability"):
    st.write("### Prediction Probability")
    st.write(f"Probability of Staying: {(1 - prediction_proba):.2f}")
    st.write(f"Probability of Leaving: {prediction_proba:.2f}")
