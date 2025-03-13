import pandas as pd
import joblib
import streamlit as st

model = joblib.load("financial_inclusion_rfc.pkl")
le  = joblib.load("financial_inclusion_le.pkl")

country = st.selectbox("Country", options=['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
location_type = st.selectbox("Location Type", options=['Rural', 'Urban'])
cellphone_access = st.selectbox("Cellphone Access", options=['Yes', 'No'])
household_size = st.number_input("Household Size (1 - 21)", min_value=1, max_value=21)
age_of_respondent = st.number_input("Age Of Respondent (16 - 100)", min_value=16, max_value=100)
gender_of_respondent = st.selectbox("Gender Of Respondent", options=['Female', 'Male'])
education_level = st.selectbox("Educational Level", options=['Secondary education',
                                                              'No formal education',
                                                                'Vocational/Specialised training', 
                                                                'Primary education',
                                                                'Tertiary education', 
                                                                'Other/Dont know/RTA'])
job_type = st.selectbox("Job Type", options=['Self employed', 'Government Dependent',
       'Formally employed Private', 'Informally employed',
       'Formally employed Government', 'Farming and Fishing',
       'Remittance Dependent', 'Other Income',
       'Dont Know/Refuse to answer', 'No Income'])

input_data = {
    "country": country,
    "location_type": location_type,
    "cellphone_access": cellphone_access,
    "household_size": household_size,
    "age_of_respondent": age_of_respondent,
    "gender_of_respondent": gender_of_respondent,
    "education_level": education_level,
    "job_type": job_type,
}

def predict_fin_inc(input_data):
    input_df = pd.DataFrame([input_data])

    for col in ['country', 'bank_account', 'location_type', 'cellphone_access', 'household_size', 'age_of_respondent', 
            'gender_of_respondent', 'education_level', 'job_type']:
        if col in input_df.columns:
            input_df[col] = le.fit_transform(input_df[col])

    prediction = model.predict(input_df)
    return prediction

if st.button("Predict Financial Inclusion"):
    prediction = predict_fin_inc(input_data)

    if prediction == 1:
        st.info("This Person Is Financially Included")
    else:
        st.info("This Person Not Is Financially Included")

