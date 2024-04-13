import streamlit as st
import pickle
import numpy as np
import spacy
import pandas as pd

import locale

# Set the locale to your system's default (or a specific locale if needed)
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')



pipe = pickle.load(open('xgb.pkl','rb'))
df = pickle.load(open('df (1).pkl','rb'))
#df = pd.read_csv('Second Def dataset.csv')

st.title("Budget Forecast for Construction Project")



total_budget_changes = st.number_input('Enter Budget changes')

total_schedule_changes = st.number_input('Enter schedule changes')



estimated_days = st.number_input('Enter estimated days')

current_phase = st.selectbox('Current Phase',df['Current Phase'].unique())



if st.button('Predict Price'):
    query = np.array([current_phase, total_budget_changes, total_schedule_changes, estimated_days], dtype=object)
    query = query.reshape(1, -1)

    pred = pipe.predict(query)[0]
    rounding = round(pred, 2)
    formatted_number = locale.currency(rounding, grouping=True)

    st.title(formatted_number)
