
          
import streamlit as st
import numpy as np
import joblib

# Load your trained model
xgb_model = joblib.load('xgb_model.pkl')

# Streamlit Application
st.title("Churn Prediction App")
st.header("Enter the Features")

# Input fields for top features with descriptive labels
region_dakar = st.number_input("Region Dakar (e.g., 1 for Yes, 0 for No)", min_value=0, max_value=1, step=1)
regularity = st.number_input("Regularity (e.g., 10)", min_value=0.0)
freq_rech_sqrt = st.number_input("Frequency Recharge (Square Root, e.g., 3.5)", min_value=0.0)
zone1_transformed = st.number_input("Zone 1 Transformed (e.g., 0.75)", min_value=0.0)
zone2 = st.number_input("Zone 2 (e.g., 1 for Yes, 0 for No)", min_value=0, max_value=1, step=1)

# Define the remaining features and their default values
all_features = ['REVENUE', 'FREQUENCE', 'ORANGE', 'ZONE2', 'REGULARITY', 'FREQ_TOP_PACK',
                'MONTANT_log', 'DATA_VOLUME_log', 'FREQUENCE_RECH_sqrt', 'ON_NET_boxcox',
                'ZONE1_transformed', 'TENURE_NUMERIC', 'REGION_DAKAR', 'REGION_DIOURBEL',
                'REGION_FATICK', 'REGION_KAFFRINE', 'REGION_KAOLACK', 'REGION_KEDOUGOU',
                'REGION_KOLDA', 'REGION_LOUGA', 'REGION_MATAM', 'REGION_SAINT-LOUIS',
                'REGION_SEDHIOU', 'REGION_TAMBACOUNDA', 'REGION_THIES', 'REGION_ZIGUINCHOR']

# Exclude top features
remaining_features = [feature for feature in all_features if feature not in ['REGION_DAKAR', 'REGULARITY', 'FREQUENCE_RECH_sqrt', 'ZONE1_transformed', 'ZONE2']]
default_values = {feature: 0 for feature in remaining_features}

# Button to make predictions
if st.button("Predict"):
    # Create a feature array
    inputs = [
        region_dakar,
        regularity,
        freq_rech_sqrt,
        zone1_transformed,
        zone2,
        *default_values.values()  # Add default values for the remaining features
    ]
    feature_array = np.array(inputs).reshape(1, -1)
    
    # Make prediction
    prediction = xgb_model.predict(feature_array)
    probability = xgb_model.predict_proba(feature_array)[0][1]
    
    # Display results
    st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    st.write(f"Churn Probability: {probability:.2f}")

