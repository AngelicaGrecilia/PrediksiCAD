#benar
import streamlit as st
from streamlit_option_menu import option_menu
import form
from numerize import numerize
import query 
from query import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os 
import pickle
import plotly.express as px
from pathlib import Path
import streamlit_authenticator as stauth


#user authenticator
# name = ["grecilia"]
# username = ["greciliahappy"]

# file_path = Path(__file__).parent / "hashed_pw.pkl"
# with file_path.open("rb") as file:
#     hashed_passwords = pickle.load(file)

# authenticator = stauth.Authenticate(name, username, hashed_passwords,
#     "password", cookie_expiry_days=30)

# name, authentication_status, username = authenticator.login("main")

# if authentication_status == False:
#     st.error("Username/password is incorrect")

# if authentication_status == None:
#     st.warning("Please enter your username and password")

# if authentication_status:

def data_analysis():
    st.header("Analysis of Patient Medical Records")
    #fetch data
    result=view_all_data_baru()
    df=pd.DataFrame(result,columns=["Age","Sex","Chest_Pain","Resting_Blood","Cholesterol","Blood_Sugar","Ekg","Heart_Rate","Angina","Oldpeak","Slope","Target","id"])
    #st.dataframe(df)
    #sidebar
    #st.sidebar.image("data/heartfailure.png")
    #switcher
    col1, col2 = st.columns(2)
    with col1:
        sex=st.multiselect(
            "Select Sex",
            options=df["Sex"].unique(),
            default=df["Sex"].unique(),
        )

    with col2:
        blood=st.multiselect(
            "Blood Blood Sugar",
            options=df["Blood_Sugar"].unique(),
            default=df["Blood_Sugar"].unique(),
        )
    col3, col4 = st.columns(2)
    with col3:
        hasilangina=st.multiselect(
            "Select angina",
            options=df["Angina"].unique(),
            default=df["Angina"].unique(),
        )
    with col4:
        target=st.multiselect(
            "Select target",
            options=df["Target"].unique(),
            default=df["Target"].unique(),
        )
    col5, col6 = st.columns(2)
    with col5:
        hasilekg=st.multiselect(
            "Select ekg",
            options=df["Ekg"].unique(),
            default=df["Ekg"].unique(),
        )
    
    with col6:
        hasilslope=st.multiselect(
            "Select slope",
            options=df["Slope"].unique(),
            default=df["Slope"].unique(),
        )
    
    chestPain=st.multiselect(
        "Chest Pain (1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic)",
        options=df["Chest_Pain"].unique(),
        default=df["Chest_Pain"].unique(),
    )

    df_selection=df.query(
        "Sex==@sex & Target==@target & Chest_Pain==@chestPain & Blood_Sugar==@blood & Ekg==@hasilekg & Angina==@hasilangina & Slope==@hasilslope"
    )

    df_selection['Sex'] = df_selection['Sex']
    df_selection['Chest_Pain'] = df_selection['Chest_Pain']
    df_selection['Blood_Sugar'] = df_selection['Blood_Sugar']
    df_selection['Ekg'] = df_selection['Ekg']
    df_selection['Angina'] = df_selection['Angina']
    df_selection['Slope'] = df_selection['Slope']
    df_selection['Target'] = df_selection['Target']

    st.dataframe(df_selection)

    def graphs():
        #simple bar graph

        chestpain_by_target=(
            df_selection.groupby(by=["Target", "Chest_Pain"]).size().unstack(fill_value=0)
        )
        fig_chestpain=px.bar(
            chestpain_by_target,
            barmode='group',
            # x="Chest_Pain",
            # y=chestpain_by_target.index,
            orientation="h",
            title="<b> Chest pain by Target </b>",
            color_discrete_sequence=["#1E0342", "#0E46A3", "#9AC8CD", "#E1F7F5"]*len(chestpain_by_target),
            template="plotly_white",
        )

        fig_chestpain.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
        )
        st.plotly_chart(fig_chestpain)

    graphs()

def predict():
    def load_components_function(Random):
        # Memuat model dari file pickle
        try:
            # Memuat model dari file pickle
            with open(Random, "rb") as f:
                model = pickle.load(f)
            with open('StandardScaler', "rb") as f:
                scaler = pickle.load(f)                
            return scaler, model
        except Exception as e:
            st.error(f"Failed to load the model. Error: {e}")
            return None
        
    st.subheader("Fill in the form!")
    # Memuat model
    DIRPATH = os.path.dirname(os.path.realpath(__file__))
    ml_core_Random = os.path.join(DIRPATH, "Random")
    print(DIRPATH)
    scaler, model = load_components_function(ml_core_Random)

#Fungsi untuk melakukan prediksi
    def prediction(model, Age, Sex, ChestPain, RestingBlood, Cholestrol, BloodSugar, ekg, heartRate, angina, oldpeak, slope):
        result=view_all_data()
        df=pd.DataFrame(result,columns=["Age","Sex","Chest_Pain","Resting_Blood","Cholesterol","Blood_Sugar","Ekg","Heart_Rate","Angina","Oldpeak","Slope","Target","id"])
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
        

        model_input = [[Age, Sex2, ChestPain2, RestingBlood, Cholestrol, BloodSugar2, ekg2, heartRate, angina2, oldpeak, slope2]]
        scaled_model_input = scaler.transform(model_input)
        print(model_input, scaled_model_input)
        # Lakukan prediksi
        predicted_output = model.predict(scaled_model_input)
        print('model output:', predicted_output)
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

#sidebar
def selection_menu():
    with st.sidebar:
        #authenticator.logout("Logout", "sidebar")
        #st.sidebar.title(f"Welcome {name}")
        selected = option_menu(
            menu_title= None, #required
            options= ["Analisis", "Prediksi"], #required
            default_index=0, #optional
            # orientation= "horizontal"
        )

    if selected == "Analisis":
        data_analysis()
    if selected == "Prediksi":
        predict()


def main():
    selection_menu()

if __name__ == "__main__":
    main()

