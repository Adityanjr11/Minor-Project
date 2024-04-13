import streamlit as st
import pickle
import numpy as np

import pandas as pd

import locale

# Set the locale to your system's default (or a specific locale if needed)
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')



pipe = pickle.load(open('pipelineMajor.pkl','rb'))
df = pickle.load(open('new_df.pkl','rb'))
#df = pd.read_csv('new_df.csv')

st.title("Budget Forecast for Construction Project")

category = st.selectbox('Category',list(df['Category'].unique()), index = None, placeholder="Select a category...")

current_phase = st.selectbox('Current Phase',list(df['Current Phase'].unique()), index = None, placeholder='Select the current phase...')

total_budget_changes = st.number_input('Enter Budget changes')

total_schedule_changes = st.number_input('Enter schedule changes')

estimated_days = st.number_input('Enter estimated days')


if st.button('Predict Price'):
    query = np.array([category, current_phase, total_budget_changes, total_schedule_changes, estimated_days], dtype=object)
    query = query.reshape(1, -1)

    pred = pipe.predict(query)[0]
    rounding = round(pred, 2)
    formatted_number = locale.currency(rounding, grouping=True)

    st.title(formatted_number)
