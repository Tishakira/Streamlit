
          
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load your trained model
xgb_model = joblib.load('xgb_modelss.pkl')

st.title('Financial Inclusion Prediction')
st.write("Predict whether an individual is likely to have a bank account.")

# Input fields for the selected features
st.write("Please enter the values for the following features:")

# Replace placeholder names with the selected features
cellphone_access_No = st.radio(
    'Do you have cellphone access?', 
    ['Yes', 'No'], 
    index=0
)
education_level_No_formal_education= st.radio(
    'Do you have formal education?', 
    ['Yes', 'No'], 
    index=1
)

education_level_Primary_education = st.radio(
    'Do you have primary education?', 
    ['Yes', 'No'], 
    index=1
)
job_type_Formally_employed_Government = st.radio(
    'Are you formally employed in the government sector?', 
    ['Yes', 'No'], 
    index=1
)
job_type_Formally_employed_Private = st.radio(
    'Are you formally employed in the private sector?', 
    ['Yes', 'No'], 
    index=1
)

# Convert categorical inputs into numerical values
cellphone_access_No = 0 if cellphone_access_No  == 'No' else 1
education_level_No_formal_education = 0 if education_level_No_formal_education == 'No' else 1
education_level_Primary_education = 0 if education_level_Primary_education == 'No' else 1
job_type_Formally_employed_Government = 0 if job_type_Formally_employed_Government == 'No' else 1
job_type_Formally_employed_Private = 0 if job_type_Formally_employed_Private == 'No' else 1

data ={'cellphone_access_No': cellphone_access_No, 
       'education_level_No formal education':education_level_No_formal_education,
       'education_level_Primary education':education_level_Primary_education,
       'job_type_Formally employed Government': job_type_Formally_employed_Government,
       'job_type_Formally employed Private' : job_type_Formally_employed_Private}

features = pd.DataFrame(data, index=[0])

# Create a button for prediction
if st.button('Predict'):
    # Make prediction
    prediction = xgb_model.predict(features)
    prediction_proba = xgb_model.predict_proba(features)
    
    # Display the result
    st.write(f"Prediction: {'Has Bank Account' if prediction[0] == 1 else 'No Bank Account'}")
    st.write(f"Probability of Having a Bank Account: {prediction_proba[0][1]:.2f}")

    
