import streamlit as st
import pickle
import numpy as np
import spacy


import locale

# Set the locale to your system's default (or a specific locale if needed)
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

nlp = spacy.load('en_core_web_sm')

pipe = pickle.load(open('etr.pkl','rb'))

st.title("Budget Forecast for Construction Project")


def encode(sentence):
    doc = nlp(sentence)
    sentence_mean_embedding = doc.vector.sum()

    return sentence_mean_embedding


total_budget_changes = st.number_input('Enter Budget changes')

total_schedule_changes = st.number_input('Enter schedule changes')

description = st.text_area('Project description')

estimated_days = st.number_input('Enter estimated days')


def remove_punct_stop(description1):
    doc = nlp(description1)
    filtered_text = ''
    for token in doc:
        filtered_text = ' '.join(token.text for token in doc if not (token.is_stop or token.is_punct))

    return filtered_text


description = encode(remove_punct_stop((description)))


if st.button('Predict Budget'):
    query = np.array([total_budget_changes, total_schedule_changes, estimated_days, description])
    query = query.reshape(1, -1)

    pred = pipe.predict(query)[0]
    rounding = round(pred, 2)
    formatted_number = locale.currency(rounding, grouping=True)

    st.title(formatted_number)
