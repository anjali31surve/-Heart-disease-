import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# Load trained model and scaler
classifier = joblib.load('knn_model.joblib')  # Change file name if using different model
scaler = joblib.load('scaler.joblib')

# Function to predict heart disease
def predict_heart(d):
    # Convert input dictionary to DataFrame
    sample_df = pd.DataFrame([d])

    # Impute any missing values (for robustness)
    imputer = SimpleImputer(strategy='median')
    sample_df = pd.DataFrame(imputer.fit_transform(sample_df), columns=sample_df.columns)

    # Scale the input
    scaled = scaler.transform(sample_df)

    # Predict and get probability
    pred = classifier.predict(scaled)[0]
    prob = np.max(classifier.predict_proba(scaled)[0])
    return pred, prob

# Streamlit app UI
st.title(" Heart Disease Prediction")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=52)
sex = st.selectbox("Sex", ["Male", "Female"])
sex_val = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], index=0)
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=125)
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=212)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1], index=0)
restecg = st.selectbox("Resting ECG results (restecg)", [0, 1, 2], index=0)
thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=168)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1], index=0)
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of peak exercise ST segment (slope)", [0, 1, 2], index=2)
ca = st.selectbox("Number of major vessels colored by fluoroscopy (ca)", [0, 1, 2, 3], index=0)
thal = st.selectbox("Thalassemia (thal)", [1, 2, 3], index=2)

# Prediction trigger
if st.button("🔍 Predict Heart Disease"):
    input_data = {
        'age': age,
        'sex': sex_val,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

try:
    pred, prob = predict_heart(input_data)
    if pred == 1:
        st.error(f" Prediction: Has Heart Disease\nConfidence: {round(prob * 100, 2)}%")
    else:
        st.success(f" Prediction: No Heart Disease\nConfidence: {round(prob * 100, 2)}%")
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
