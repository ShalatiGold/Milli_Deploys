import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the best model and preprocessor
model = joblib.load('C:/Users/User/Documents/models/best_model.pkl')
preprocessor = joblib.load('C:/Users/User/Documents/models/preprocessor.pkl')

# Define the input fields for the user
st.title('Heart Disease Prediction System')
st.write("Please enter the Patient's details below to proceed with the prediction")

age = st.number_input('Age', min_value=0, max_value=120, value=25)
sex = st.selectbox('Sex (1=Male, 0=Female)', options=[0, 1])
cp = st.selectbox('Chest Pain Type (0=Typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic)', options=[0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (in mm Hg)', min_value=0, max_value=300, value=120)
chol = st.number_input('Serum Cholesterol (in mg/dl)', min_value=0, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', options=[0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results (0=Normal, 1=Abnormal, 2=Ventricular Hypertrophy)', options=[0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved ', min_value=0, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina (1=Yes, 0=No)', options=[0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise relative to rest', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox('Slope of the Peak Exercise ST Segment (0=Upsloping, 1=Flat, 2=Downslopping)', options=[0, 1, 2])
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy ', options=[0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia (1=Normal, 2=Fixed Defect, 3 =Reversible Defect, 0=Unknown)', options=[0, 1, 2, 3])

# Create a button for prediction
if st.button('Predict'):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    # Apply the same preprocessing to the input data
    input_data_transformed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_transformed)
    prediction_proba = model.predict_proba(input_data_transformed)

    # Display the prediction result
    if prediction[0] == 1:
        #st.error(f"The patient is likely to have heart disease. (Probability: {prediction_proba[0][1]:.2f})")
        st.error(f"The Patient has a of heart disease. ")
    else:
        #st.success(f"The patient is unlikely to have heart disease. (Probability: {prediction_proba[0][0]:.2f})")
        st.error(f"The Patient does not have a  of heart disease. ")
