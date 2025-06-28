import pandas as pd
import streamlit as st
import numpy as np
import sklearn
import pickle
from sklearn.preprocessing import LabelEncoder,StandardScaler
Label_encoder=LabelEncoder()
scaler=StandardScaler()
model=pickle.load(open('Customer_churn_analysis.pkl','rb'))
st.title("Customer Churn Prediction")
gender=st.selectbox("Enter your gender",options=['Male','Fenale'])
SeniorCitizen=st.text_input("Enter O or 1")
Partner=st.selectbox("Do you have a partner?",options=['Yes','No'],key="partner_key")
Dependents=st.selectbox("Are you dependent",options=['Yes','No'],key="Dependents_key")
tenure=st.text_input("Enter your tenure")
PhoneService=st.selectbox("Do you have phone service?",options=['Yes','No'],key="phoneservice_key")
MultipleLines=st.selectbox("Do you have a MutplineLines service?",options=['Yes','No','No phone service'],key="MutipleLines_key")
InternetService=st.selectbox("Do you have a InternetService?",options=['Fibre optic','DSL','No'],key="InternetService_key")
OnlineSecurity=st.selectbox("Do you have OnlineSecurity?",options=['Yes','No'],key="OnlineSecurity_key")
OnlineBackup=st.selectbox("Do you have OnlineBackup?",options=['Yes','No'],key="OnlineBackup_key")
DeviceProtection=st.selectbox("Do you have DeviceProtection?",options=['Yes','No'],key="DeviceProtection_key")
TechSupport=st.selectbox("Do you have a TechSupport?",options=['Yes','No'],key="TechSupport_key")
StreamingTV=st.selectbox("Do you have a StreamingTV?",options=['Yes','No'],key="StreamingTV_key")
StreamingMovies=st.selectbox("Do you have StreamingMovies?",options=['Yes','No'],key="StreamingMovies_key")
Contract=st.selectbox("Do you have contract?",options=['Month-to-month','Two year','One Year'],key="contract_key")
PaperlessBilling=st.selectbox("Do you have a PaperlessBilling?",options=['Yes','No'],key="PaperlessBilling_key")
PaymentMethod=st.selectbox("Do you have a PaymentMethod?",options=['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'],key="PaymentMethod_key")
MonthlyCharges=st.text_input("Enter your MonthlyCharge")
TotalCharges=st.text_input("Enter your TotalCharges")
#=============================================================================================================================================================================================================================
def predictive(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
MonthlyCharges, TotalCharges):

    data = {
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    }

    df1 = pd.DataFrame(data)

    # Encoding categorical data
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService','MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract', 'PaperlessBilling', 'PaymentMethod']

    for column in categorical_columns:
        df1[column] = Label_encoder.fit_transform(df1[column])

    df1= scaler.fit_transform(df1)
    result =model.predict(df1).reshape(1,-1)

    return result[0]



#==============================================================================================================================================================================================================================
if st.button("predict"):
    result = predictive(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
                        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                        StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges)
    if result==0:
        st.write("Not churn")
    else:
        st.write("Churn")
