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
sex = st.selectbox("Sex", options=["M", "F"])

last_performance_rating = st.selectbox(
    "Last Performance Rating", 
    options=["A", "B", "C", "PIP", "S"]
)

job_title = st.selectbox(
    "Job Title", 
    options=[
        "Senior Engineer", "Staff", "Assistant Engineer", 
        "Technique Leader", "Engineer", "Senior Staff", "Manager"
    ]
)

dept_name = st.selectbox(
    "Department", 
    options=[
        "Development", "Sales", "Production", "Human Resources", 
        "Research", "Quality Management", "Customer Service", 
        "Marketing", "Finance"
    ]
)

no_of_projects = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
salary = st.number_input("Salary", min_value=40000, max_value=130000, value=50000)

# ✅ Encode categorical data (for model input)
sex_encoded = 1 if sex == "M" else 0
last_performance_rating_map = {"A": 0, "B": 1, "C": 2, "PIP": 3, "S": 4}
last_performance_rating_encoded = last_performance_rating_map[last_performance_rating]

job_title_map = {
    "Senior Engineer": 0, "Staff": 1, "Assistant Engineer": 2,
    "Technique Leader": 3, "Engineer": 4, "Senior Staff": 5, "Manager": 6
}
job_title_encoded = job_title_map[job_title]

dept_name_map = {
    "Development": 0, "Sales": 1, "Production": 2, "Human Resources": 3,
    "Research": 4, "Quality Management": 5, "Customer Service": 6,
    "Marketing": 7, "Finance": 8
}
dept_name_encoded = dept_name_map[dept_name]

# ✅ Prepare the input data (Reorder to match training order)
input_data = pd.DataFrame(
    {
        "sex": [sex_encoded],
        "last_performance_rating": [last_performance_rating_encoded],
        "job_title": [job_title_encoded],
        "dept_name": [dept_name_encoded],
        "no_of_projects": [no_of_projects],
        "salary": [salary],
    }
)

# ✅ Match the order of features used during model training
input_data = input_data[['sex', 'last_performance_rating', 'job_title', 'dept_name', 'no_of_projects', 'salary']]

# ✅ Scale numeric features
input_data[["no_of_projects", "salary"]] = scaler.transform(input_data[["no_of_projects", "salary"]])

# ✅ Prediction
if st.button("Predict"):
    prediction = model.predict(input_data, validate_features=False)[0]
    prediction_proba = model.predict_proba(input_data)[:, 1][0]
    
    if prediction == 1:
        st.error(f"🚨 The employee is likely to leave the company (Confidence: {prediction_proba:.2f})")
    else:
        st.success(f"✅ The employee is likely to stay with the company (Confidence: {prediction_proba:.2f})")

# ✅ Feature importance plot
if st.checkbox("Show Feature Importance"):
    feature_importance = model.feature_importances_
    feature_names = ['sex', 'last_performance_rating', 'job_title', 'dept_name', 'no_of_projects', 'salary']
    
    st.write("### Feature Importance")
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    
    st.bar_chart(importance_df.set_index("Feature"))

# ✅ Prediction probability plot
if st.checkbox("Show Prediction Probability"):
    st.write("### Prediction Probability")
    st.write(f"Probability of Staying: {(1 - prediction_proba):.2f}")
    st.write(f"Probability of Leaving: {prediction_proba:.2f}")
