import pickle
import streamlit as st
import pandas as pd
import xgboost
import numpy as np

# Load the saved XGBoost model
loaded_model = xgboost.Booster()
loaded_model.load_model('xgb_model.bin')

# Page title
st.title('Heart Attack Prediction using ML')

# Show center image
st.image("https://media.tenor.com/91scJf-xrKEAAAAi/emoji-coraz%C3%B3n-humano.gif", width=200, caption="Healthy Heart", use_column_width=True)

# Input fields
age = st.number_input('Enter age', step=1)
sex = st.selectbox('Enter sex', ('Male', 'Female'))
sex = 1 if sex == 'Male' else 0

st.subheader("Chest Pain Type")
st.info("0: typical angina | 1: atypical angina | 2: non-anginal pain | 3: asymptomatic")
cp = st.selectbox('Select Chest Pain type', (0, 1, 2, 3))

trestbps = st.number_input('Enter resting blood pressure (mm Hg)', step=1)
chol = st.number_input('Enter cholesterol (mg/dl)', step=1)

fbs = st.selectbox('Is fasting blood sugar > 120 mg/dl?', ('Yes', 'No'))
fbs = 1 if fbs == 'Yes' else 0

st.subheader("Resting ECG Results")
st.info("0: normal | 1: ST-T abnormality | 2: LV hypertrophy")
restecg = st.selectbox('Enter Resting ECG value', (0, 1, 2))

thalach = st.number_input("Enter max heart rate achieved", step=1)

exang = st.selectbox('Exercise induced angina?', ('Yes', 'No'))
exang = 1 if exang == 'Yes' else 0

oldpeak = st.number_input('Enter oldpeak value (ST depression)')

st.subheader("Slope of peak exercise ST segment")
st.info("0: downsloping | 1: flat | 2: upsloping")
slp = st.selectbox('Enter slope value', (0, 1, 2))

ca = st.selectbox('Number of major vessels (0–3)', (0, 1, 2, 3))
thal= st.selectbox('Enter thalassemia value', (0, 1, 2, 3))

# Check for required inputs
features_values = {'age': age, 'trestbps': trestbps, 'chol': chol, 'thalach': thalach, 'oldpeak': oldpeak}

if st.button('Predict'):
    if any(value == 0 or value == 0.00 for value in features_values.values()):
        st.warning('Please input all the details.')
    else:
        data_1 = pd.DataFrame({
            'thal': [thal],
            'ca': [ca],
            'cp': [cp],
            'oldpeak': [oldpeak],
            'exang': [exang],
            'chol': [chol],
            'thalach': [thalach]
        })

        dtest = xgboost.DMatrix(data_1)
        prediction = loaded_model.predict(dtest)
        prediction = np.where(prediction >= 0.5, 1, 0)

        if prediction == 0:
            st.success("✅ Patient has no risk of Heart Attack")
        else:
            st.error("⚠️ Patient has risk of Heart Attack")
