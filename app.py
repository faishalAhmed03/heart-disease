import streamlit as st
import joblib
import numpy as np
import pandas as pd

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load the trained model
model = joblib.load('xgb_model.pkl')

# Streamlit app title
st.title('Heart Disease Prediction App')

# Add a sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 80, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 100, 400, 200)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 200, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ('Yes', 'No'))
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 5.0, 1.0)
    slope = st.sidebar.slider('Slope of the Peak Exercise ST Segment', 0, 2, 1)
    ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy (0-3)', 0, 3, 0)
    thal = st.sidebar.slider('Thalassemia (1-3)', 1, 3, 2)
    
    # Convert categorical variables to numerical
    sex = 1 if sex == 'Male' else 0
    exang = 1 if exang == 'Yes' else 0
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Collect user input
input_df = user_input_features()

# Display user input parameters
st.subheader('User Input Parameters')
st.write(input_df)

# Make predictions only when input is provided
if st.sidebar.button("Predict"):
    # Make predictions
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display prediction probability
    st.subheader('Prediction Probability')
    st.write(prediction_proba)
    
    # Display prediction
    st.subheader('Prediction')
    heart_disease_status = np.array(['Person does not have heart disease.', 'Person has heart disease.'])
    st.markdown(f"**{heart_disease_status[prediction][0]}**")


