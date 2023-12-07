import streamlit as st
import pickle
import numpy as np
import spacy
import sklearn


nlp = spacy.load('en_core_web_sm')

pipe = pickle.load(open('model.pkl','rb'))

st.title("Budget Forecast Prediction")

def encode(sentence):
    doc = nlp(sentence)
    sentence_mean_embedding = doc.vector.mean()

    return sentence_mean_embedding

total_budget_changes = st.number_input('Enter Budget changes')

total_schedule_changes = st.number_input('Enter schedule changes')

description = st.text_area('Project description')


description = encode(description)

if st.button('Predict Price'):
    query = np.array([total_budget_changes, total_schedule_changes, description])
    query = query.reshape(1, -1)
    #st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
    st.title("$" + str(int(pipe.predict(query)[0])))










