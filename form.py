import streamlit as st
import os 
import pickle


st.set_page_config (page_title="Coronary Artery Disease Prediction", layout="wide")
#st.subheader("Welcome to the website-based Coronary Artery Disease detection system")
st.sidebar.title("Navigation")
st.subheader("Fill in the form")

DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_Random = os.path.join(DIRPATH, "RandomForest")

try:
    with open(ml_core_Random, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load the model. Error: {e}")

def prediction(model, Age, Sex, ChestPain, RestingBlood, Cholestrol, BloodSugar, ekg, heartRate, angina, oldpeak, slope):
    if (Sex == 'Male'):
        Sex2 = 1
    else:
        Sex2 = 0
    
    if (ChestPain == 'Typical angina'):
        ChestPain2 = 1
    elif (ChestPain == 'Atypical angina'):
        ChestPain2 = 2
    elif (ChestPain == 'Non-angina pain'):
        ChestPain2 = 3
    else:
        ChestPain2 = 4
        
    if (BloodSugar == 'Greater than 120 mg/dl'):
        BloodSugar2 = 1
    else:
        BloodSugar2 = 0
        
    if (angina == 'Yes'):
        angina2 = 1
    else:
        angina2 = 0
        
    if (ekg == 'Normal'):
        ekg2 = 0
    elif (ekg == 'ST_T wave abnormality'):
        ekg2 = 1
    else:
        ekg2 = 2
        
    if (slope == 'Normal'):
        slope2 = 0
    elif (slope == 'Uplsloping'):
        slope2 = 1
    elif (slope == 'Flat'):
        slope2 = 2
    else:
        slope2 = 3
    
    print([Age, Sex2, ChestPain2, RestingBlood, Cholestrol, BloodSugar2, ekg2, heartRate, angina2, oldpeak, slope2])
    # Lakukan prediksi
    predicted_output = model.predict([[Age, Sex2, ChestPain2, RestingBlood, Cholestrol, BloodSugar2, ekg2, heartRate, angina2, oldpeak, slope2]])
    return predicted_output[0]  # Ambil hasil prediksi dari array

with st.form(key="information", clear_on_submit=True):
    Age = st.number_input("Age", 0, None)
    Sex = st.selectbox('Sex', ['Female', 'Male'])
    ChestPain = st.selectbox("Chest Pain Type", ['Typical angina', 'Atypical angina', 'Non-angina pain', 'Asymptomatic'])
    RestingBlood = st.number_input("Resting Blood Pressure ", 0, None)
    Cholestrol = st.number_input("Cholestrol", 0, None)
    BloodSugar = st.selectbox("Fasting Blood Sugar", ["Greater than 120 mg/dl", "Less than 120 mg/dl"])
    ekg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST_T wave abnormality", "Showing probable or definite left ventricular hypertrophy by Estes' criteria"])
    heartRate = st.number_input("Maximum Heart Rate", 71, 202)
    angina = st.selectbox("Exercise induced angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak=ST", 0.0, None)
    slope = st.selectbox("The slope of the peak exercise ST segment", ["Normal", "Upsloping", "Flat", "Downsloping"])

    if st.form_submit_button("Predict"):
        result = prediction(model, Age, Sex, ChestPain, RestingBlood, Cholestrol, BloodSugar, ekg, heartRate, angina, oldpeak, slope)
        if result == 1:
            st.success('Chance of Heart attack')
        else:
            st.error('Not having Heart attack')