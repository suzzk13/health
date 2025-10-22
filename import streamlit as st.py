import streamlit as st
import joblib
import pandas as pd
import os

# --- 1. CONFIGURATION AND MODEL LOADING ---

# Define the model filename (Must match the file saved from Colab)
MODEL_FILENAME = 'best_health_risk_model_pipeline.pkl'

# Set page configuration
st.set_page_config(
    page_title="Health Risk Predictor Demo",
    layout="wide",
    initial_sidebar_state="auto"
)

# Load the trained pipeline (which includes the model and preprocessor)
@st.cache_resource
def load_model():
    # Check if the model file is available in the deployment environment
    if os.path.exists(MODEL_FILENAME):
        try:
            full_pipeline = joblib.load(MODEL_FILENAME)
            return full_pipeline
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
    else:
        st.error(f"FATAL: Model file '{MODEL_FILENAME}' not found. Please ensure it is uploaded to your GitHub repository.")
        st.stop()

full_pipeline = load_model()

# --- 2. STREAMLIT USER INTERFACE (UI) ---

st.title("💖 Health Disease Risk Prediction System")
st.markdown("---")
st.subheader("Enter Patient's Health and Lifestyle Metrics")

# Create columns for organized input fields
col1, col2, col3 = st.columns(3)

# Numerical Inputs (Column 1)
with col1:
    st.markdown("#### Physical Attributes")
    age = st.slider("Age (years)", 18, 90, 40)
    bmi = st.number_input("BMI (kg/m²)", 15.0, 50.0, 25.0, step=0.1)
    resting_hr = st.number_input("Resting Heart Rate (bpm)", 40, 120, 70)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 300, 180)

# Lifestyle Inputs (Column 2)
with col2:
    st.markdown("#### Lifestyle Factors")
    daily_steps = st.number_input("Daily Steps", 0, 20000, 5000)
    sleep_hours = st.slider("Sleep Hours", 3.0, 12.0, 7.0, step=0.1)
    water_intake_l = st.slider("Water Intake (Liters)", 0.5, 5.0, 2.0, step=0.1)
    calories_consumed = st.number_input("Calories Consumed (kcal)", 500, 5000, 2000)

# Categorical & BP Inputs (Column 3)
with col3:
    st.markdown("#### Health & History")
    gender = st.selectbox("Gender", ['male', 'female'])
    smoker = st.selectbox("Smoker", ['yes', 'no'])
    alcohol = st.selectbox("Alcohol Consumption", ['yes', 'no'])
    family_history = st.selectbox("Family History of Disease", ['yes', 'no'])
    systolic_bp = st.number_input("Systolic BP (mm Hg)", 80, 200, 120)
    diastolic_bp = st.number_input("Diastolic BP (mm Hg)", 50, 150, 80)


# --- 3. PREDICTION LOGIC ---

if st.button("Predict Disease Risk", type="primary"):
    # 1. Collect all inputs into a dictionary
    input_data_dict = {
        'age': [age],
        'gender': [gender],
        'bmi': [bmi],
        'daily_steps': [daily_steps],
        'sleep_hours': [sleep_hours],
        'water_intake_l': [water_intake_l],
        'calories_consumed': [calories_consumed],
        'resting_hr': [resting_hr],
        'systolic_bp': [systolic_bp],
        'diastolic_bp': [diastolic_bp],
        'cholesterol': [cholesterol],
        'smoker': [smoker],
        'alcohol': [alcohol],
        'family_history': [family_history]
    }

    # 2. Convert to DataFrame (ensuring correct column order and structure)
    input_df = pd.DataFrame(input_data_dict)

    # 3. Make prediction
    prediction = full_pipeline.predict(input_df)[0]
    
    # 4. Format and display result
    st.markdown("---")
    st.subheader("Prediction Result:")
    
    if prediction == 1:
        st.error("🚨 **Predicted Risk: HIGH**")
        st.warning("The system predicts this profile has a high risk of developing a disease. Consultation is recommended.")
    else:
        st.success("